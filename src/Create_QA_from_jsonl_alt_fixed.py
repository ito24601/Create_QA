#!/usr/bin/env python3
"""
評価・改善サイクル付き6段階エージェント処理でJSONLファイルからQ&Aペアを生成

このスクリプトは、JSONLファイルから高品質なQ&Aペアを生成するために、
6つの専門エージェントを使用します：

1. Q&A生成エージェント: 基本的な質問-回答ペアを生成
2. Q&A評価エージェント: 生成されたQ&Aの品質を評価
3. Q&A改善エージェント: 評価フィードバックに基づいてQ&Aを改善
4. ペルソナ分析エージェント: 質問者のペルソナを特定
5. カテゴリ分類エージェント: 情報カテゴリを分類
6. キーワード抽出エージェント: 関連キーワードを抽出

各エージェントは特定の専門分野に集中し、高品質な結果を提供します。
評価・改善サイクルにより、自動的にQ&Aの品質向上を図ります。
"""
import asyncio
import jsonlines
import os
import argparse
import threading
from typing import List, Set, Tuple, Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from enum import Enum

from agents import Agent, Runner # agentsモジュールからAgentとRunnerをインポート

load_dotenv("/app/.env", override=True)

# --- エージェント設定管理クラス ---
class AgentConfig:
    """各エージェントの個別設定を管理するクラス"""
    
    def __init__(self, base_model: str = "gpt-4o-mini"):
        self.base_model = base_model
        
        # 各エージェントの個別設定
        self.agents = {
            "qa_generation": {
                "model": "gpt-4o",  # 高品質が必要なのでより性能の高いモデル
                "temperature": 0.7,  # 創造性を少し高める
                "max_tokens": 1000,
                "timeout": 60
            },
            "qa_evaluation": {
                "model": "gpt-4o",  # 評価の一貫性のため高性能モデル
                "temperature": 0.3,  # 評価の一貫性のため低温度
                "max_tokens": 800,
                "timeout": 45
            },
            "qa_improvement": {
                "model": "gpt-4o",  # 改善には複雑な推論が必要
                "temperature": 0.5,  # バランスの取れた創造性
                "max_tokens": 1000,
                "timeout": 60
            },
            "persona_analysis": {
                "model": "gpt-4o-mini",  # 分類タスクなので効率的なモデル
                "temperature": 0.2,  # 分類の一貫性のため低温度
                "max_tokens": 200,
                "timeout": 30
            },
            "category_analysis": {
                "model": "gpt-4o-mini",  # 分類タスクなので効率的なモデル
                "temperature": 0.2,  # 分類の一貫性のため低温度
                "max_tokens": 200,
                "timeout": 30
            },
            "keyword_extraction": {
                "model": "gpt-4o-mini",  # キーワード抽出は効率的なモデルで十分
                "temperature": 0.1,  # キーワード抽出の一貫性のため最低温度
                "max_tokens": 300,
                "timeout": 30
            }
        }
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """指定されたエージェントの設定を取得"""
        return self.agents.get(agent_name, {
            "model": self.base_model,
            "temperature": 0.5,
            "max_tokens": 500,
            "timeout": 30
        })
    
    def set_agent_model(self, agent_name: str, model: str):
        """特定のエージェントのモデルを設定"""
        if agent_name in self.agents:
            self.agents[agent_name]["model"] = model
    
    def set_agent_temperature(self, agent_name: str, temperature: float):
        """特定のエージェントのtemperatureを設定"""
        if agent_name in self.agents:
            self.agents[agent_name]["temperature"] = temperature
    
    def set_quality_mode(self, mode: str):
        """品質モードを設定（all_premium, all_standard, balanced）"""
        if mode == "all_premium":
            # すべて最高性能モデル
            for agent_name in self.agents:
                self.agents[agent_name]["model"] = "gpt-4o"
        elif mode == "all_standard":
            # すべて標準モデル
            for agent_name in self.agents:
                self.agents[agent_name]["model"] = "gpt-4o-mini"
        elif mode == "balanced":
            # デフォルト設定（重要なタスクのみ高性能モデル）
            pass  # 既にデフォルト設定済み
    
    def print_config(self):
        """現在の設定を表示"""
        print("🔧 エージェント設定:")
        for agent_name, config in self.agents.items():
            print(f"  {agent_name}: {config['model']} (temp: {config['temperature']}, max_tokens: {config['max_tokens']})")

# グローバル設定インスタンス
agent_config = AgentConfig()

# --- データモデル ---
class EvaluationScore(str, Enum):
    EXCELLENT = "excellent"  # 90-100点
    GOOD = "good"           # 70-89点  
    FAIR = "fair"           # 50-69点
    POOR = "poor"           # 0-49点

class BasicQAPair(BaseModel):
    """基本的なQ&Aペア（メタデータなし）"""
    question: str
    answer: str
    source_url: str

class PersonaResult(BaseModel):
    """ペルソナ分析結果"""
    questioner_persona: str

class CategoryResult(BaseModel):
    """カテゴリ分析結果"""
    information_category: str

class KeywordsResult(BaseModel):
    """キーワード分析結果"""
    related_keywords: List[str]

class QAEvaluation(BaseModel):
    """Q&A評価結果"""
    overall_score: int  # 0-100点
    overall_rating: EvaluationScore
    
    # 評価項目別スコア
    source_coverage_score: int      # 元ソースに回答情報が含まれているか (0-100)
    question_specificity_score: int # 質問が十分に背景情報を含むか (0-100)
    condition_clarity_score: int    # 条件が明確に示されているか (0-100)
    
    # フィードバック
    strengths: List[str]           # 良い点
    improvement_areas: List[str]   # 改善点
    specific_suggestions: List[str] # 具体的な改善提案
    
    # 改善の必要性
    needs_improvement: bool
    improvement_priority: str  # "high", "medium", "low"

class QAPair(BaseModel):
    """完全なQ&Aペア（メタデータ付き）"""
    question: str
    answer: str
    source_url: str # JSONLの各エントリの出典を示すために流用
    questioner_persona: str # 追加: どのような人がする質問か
    information_category: str  # 追加: 情報のカテゴリ
    related_keywords: List[str] # 追加: 関連キーワード
    # 評価・改善情報
    evaluation_score: Optional[int] = None  # 最終評価スコア (0-100)
    evaluation_rating: Optional[EvaluationScore] = None  # 最終評価レベル
    improvement_iterations: Optional[int] = None  # 改善サイクル実行回数

# --- エージェント1: Q&A生成専用 ---
async def generate_basic_qa(
    source_identifier: str, # URLやファイル名など、コンテンツの出典
    text_content: str,
    existing_qa_for_source_display: List[str],
    model_name: str,
    attempt_number: int  # 何回目の試行かを明示
) -> Optional[BasicQAPair]:
    """
    基本的なQ&Aペアのみを生成（メタデータなし）
    """
    if not existing_qa_for_source_display:
        existing_qa_instructions_segment = "There are currently no existing Q&A pairs for this source."
    else:
        existing_qa_str = "\\n".join(existing_qa_for_source_display)
        existing_qa_instructions_segment = (
            f"The following Q&A pairs already exist for this source ({source_identifier}):\\n"
            f"Please generate a NEW Q&A pair that covers different aspects or provides different perspectives.\\n"
            f"---Existing Q&A Start---\\n"
            f"{existing_qa_str}\\n"
            f"---Existing Q&A End---"
        )

    # 個別エージェント設定を取得
    config = agent_config.get_agent_config("qa_generation")
    
    qa_generation_agent = Agent(
        name="QA Generation Specialist",
        instructions=(
            "You are a specialized Q&A generation assistant focused solely on creating high-quality question-answer pairs.\\n"
            f"1. Analyze the provided text content from: {source_identifier} (likely a life insurance company's webpage).\\n"
            f"2. Text content: \\\\n---TEXT CONTENT BEGIN---\\\\n{text_content}\\\\n---TEXT CONTENT END---\\\\n"
            f"3. {existing_qa_instructions_segment}\\\\n"
            f"4. Generate ONE high-quality question-answer pair. Focus on:\\n"
            "   - Creating a natural, specific question someone would actually ask\\n"
            "   - If the answer varies based on conditions (age, gender, health status, contract details, timing, etc.), make the question specify those conditions clearly\\n"
            "   - If the answer differs by insurance product, include the specific product name in the question\\n"
            "   - For example: instead of '保険金はいくらもらえますか？' ask '30歳男性がちゃんと応える医療保険EVERに加入した場合、入院給付金はいくらもらえますか？'\\n"
            "   - Another example: instead of '保険料の支払い方法は？' ask 'アフラックのがん保険フォルテの保険料支払い方法にはどのような選択肢がありますか？'\\n"
            "   - Providing a comprehensive, self-contained answer that addresses the specific conditions and products mentioned in the question\\n"
            "   - Avoiding generic or overly broad questions that could have multiple different answers\\n"
            "   - Including relevant details and context\\n"
            f"5. This is attempt #{attempt_number}, so try to find a unique angle or aspect not covered before.\\n"
            "6. The question and answer MUST be in Japanese.\\n"
            "7. The answer should be self-contained and directly address the question. Avoid answers that primarily redirect the user elsewhere.\\n"
            "8. Return exactly ONE BasicQAPair object with question, answer, and source_url fields only.\\n"
            f"9. The source_url must be exactly: '{source_identifier}'"
        ),
        output_type=BasicQAPair,
        model=config["model"],
    )

    try:
        result = await Runner.run(qa_generation_agent, input=f"Generate one high-quality Q&A for content from {source_identifier}")
        
        if result.final_output:
            qa = result.final_output
            # source_urlの修正
            if qa.source_url != source_identifier:
                qa_dict = qa.model_dump()
                qa_dict["source_url"] = source_identifier
                return BasicQAPair(**qa_dict)
            return qa
    except Exception as e:
        print(f"    ❌ Q&A生成エラー: {e}")
    
    return None

# --- エージェント2: ペルソナ分析専用 ---
async def generate_persona(
    basic_qa: BasicQAPair,
    source_identifier: str,
    text_content: str,
    model_name: str
) -> Optional[PersonaResult]:
    """
    Q&Aペルソナ分析専用エージェント
    """
    # 個別エージェント設定を取得
    config = agent_config.get_agent_config("persona_analysis")
    
    persona_agent = Agent(
        name="Persona Analysis Specialist",
        instructions=(
            "You are a specialized persona analysis assistant focused on identifying who would ask specific questions.\\n"
            f"1. Analyze the provided Q&A pair and its source context from: {source_identifier}\\n"
            f"2. Source context: \\n---SOURCE CONTENT---\\n{text_content[:1000]}...\\n---END SOURCE CONTENT---\\n"
            f"3. Q&A pair to analyze:\\n"
            f"   Question: {basic_qa.question}\\n"
            f"   Answer: {basic_qa.answer}\\n"
            "4. Determine the questioner_persona - who would likely ask this specific question?\\n"
            "5. Consider life insurance website visitors and their motivations:\\n"
            "   - '契約検討中の顧客': Someone considering purchasing insurance\\n"
            "   - '既契約者': Existing policyholders with questions about their coverage\\n"
            "   - '保険金受取人': Beneficiaries or claimants\\n"
            "   - '就職活動中の学生': Job-seeking students interested in company benefits\\n"
            "   - '一般的な情報収集者': General information seekers\\n"
            "   - '保険料を検討中の顧客': Customers comparing premium costs\\n"
            "   - '健康に関心がある人': Health-conscious individuals\\n"
            "   - '介護に関心がある人': People interested in long-term care\\n"
            "6. Choose the most appropriate persona based on the question's content and intent.\\n"
            "7. The questioner_persona must be in Japanese.\\n"
            "8. Return exactly ONE PersonaResult object with questioner_persona field."
        ),
        output_type=PersonaResult,
        model=config["model"],
    )

    try:
        result = await Runner.run(persona_agent, input=f"Analyze persona for Q&A: {basic_qa.question}")
        return result.final_output if result.final_output else None
    except Exception as e:
        print(f"    ⚠️ ペルソナ分析エラー: {e}")
        return None

# --- エージェント3: カテゴリ分類専用 ---
async def generate_category(
    basic_qa: BasicQAPair,
    source_identifier: str,
    text_content: str,
    model_name: str
) -> Optional[CategoryResult]:
    """
    Q&Aカテゴリ分類専用エージェント
    """
    # 個別エージェント設定を取得
    config = agent_config.get_agent_config("category_analysis")
    
    category_agent = Agent(
        name="Category Classification Specialist",
        instructions=(
            "You are a specialized category classification assistant focused on categorizing insurance-related Q&A pairs.\\n"
            f"1. Analyze the provided Q&A pair and its source context from: {source_identifier}\\n"
            f"2. Source context: \\n---SOURCE CONTENT---\\n{text_content[:1000]}...\\n---END SOURCE CONTENT---\\n"
            f"3. Q&A pair to analyze:\\n"
            f"   Question: {basic_qa.question}\\n"
            f"   Answer: {basic_qa.answer}\\n"
            "4. Determine the information_category - what type of information does this Q&A provide?\\n"
            "5. Choose from these standard insurance information categories:\\n"
            "   - '契約手続き': Contract procedures, applications, policy changes\\n"
            "   - '保障内容': Coverage details, benefits, policy features\\n"
            "   - '保険金請求': Claims procedures, benefit payments\\n"
            "   - '商品比較': Product comparisons, plan differences\\n"
            "   - '税金・控除': Tax implications, deductions\\n"
            "   - '健康増進サービス': Wellness services, health programs\\n"
            "   - '会社情報': Company information, corporate data\\n"
            "   - '保険料シミュレーション': Premium calculations, cost estimates\\n"
            "   - '相談方法': Consultation methods, contact information\\n"
            "6. Select the most appropriate single category based on the primary focus of the Q&A.\\n"
            "7. The information_category must be in Japanese.\\n"
            "8. Return exactly ONE CategoryResult object with information_category field."
        ),
        output_type=CategoryResult,
        model=config["model"],
    )

    try:
        result = await Runner.run(category_agent, input=f"Classify category for Q&A: {basic_qa.question}")
        return result.final_output if result.final_output else None
    except Exception as e:
        print(f"    ⚠️ カテゴリ分類エラー: {e}")
        return None

# --- エージェント4: キーワード抽出専用 ---
async def generate_keywords(
    basic_qa: BasicQAPair,
    source_identifier: str,
    text_content: str,
    model_name: str
) -> Optional[KeywordsResult]:
    """
    Q&Aキーワード抽出専用エージェント
    """
    # 個別エージェント設定を取得
    config = agent_config.get_agent_config("keyword_extraction")
    
    keywords_agent = Agent(
        name="Keywords Extraction Specialist",
        instructions=(
            "You are a specialized keywords extraction assistant focused on identifying relevant search terms for insurance Q&A pairs.\\n"
            f"1. Analyze the provided Q&A pair and its source context from: {source_identifier}\\n"
            f"2. Source context: \\n---SOURCE CONTENT---\\n{text_content[:1000]}...\\n---END SOURCE CONTENT---\\n"
            f"3. Q&A pair to analyze:\\n"
            f"   Question: {basic_qa.question}\\n"
            f"   Answer: {basic_qa.answer}\\n"
            "4. Extract 3-5 related_keywords that represent the core topics and concepts in this Q&A.\\n"
            "5. Keywords should be:\\n"
            "   - Relevant to the insurance industry\\n"
            "   - Specific to the content of this Q&A\\n"
            "   - Useful for search and categorization\\n"
            "   - Include product names, procedures, or specific terms mentioned\\n"
            "   - Mix of general and specific terms\\n"
            "6. Example keywords for different topics:\\n"
            "   - For medical insurance: ['医療保険', '入院給付金', '通院', '健康診断']\\n"
            "   - For cancer insurance: ['がん保険', '診断給付金', '治療費', '先進医療']\\n"
            "   - For claims: ['保険金請求', '給付金', '必要書類', '手続き']\\n"
            "7. All keywords must be in Japanese.\\n"
            "8. Return exactly ONE KeywordsResult object with related_keywords list (3-5 items)."
        ),
        output_type=KeywordsResult,
        model=config["model"],
    )

    try:
        result = await Runner.run(keywords_agent, input=f"Extract keywords for Q&A: {basic_qa.question}")
        return result.final_output if result.final_output else None
    except Exception as e:
        print(f"    ⚠️ キーワード抽出エラー: {e}")
        return None

# --- エージェント5: Q&A評価専用 ---
async def evaluate_qa_quality(
    basic_qa: BasicQAPair,
    source_identifier: str,
    text_content: str,
    model_name: str
) -> Optional[QAEvaluation]:
    """
    生成されたQ&Aの品質を評価する専用エージェント
    """
    # 個別エージェント設定を取得
    config = agent_config.get_agent_config("qa_evaluation")
    
    evaluation_agent = Agent(
        name="QA Quality Evaluator",
        instructions=(
            "You are a specialized Q&A quality evaluation assistant focused on assessing insurance-related Q&A pairs.\\n"
            f"1. Analyze the provided Q&A pair and its source context from: {source_identifier}\\n"
            f"2. Source text content: \\n---SOURCE CONTENT BEGIN---\\n{text_content}\\n---SOURCE CONTENT END---\\n"
            f"3. Q&A pair to evaluate:\\n"
            f"   Question: {basic_qa.question}\\n"
            f"   Answer: {basic_qa.answer}\\n"
            "4. Evaluate based on these key criteria:\\n"
            "\\n"
            "**A. Source Coverage (0-100 points):**\\n"
            "   - Does the answer information exist in the source content?\\n"
            "   - Is the answer based on factual information from the source?\\n"
            "   - Are there any claims in the answer not supported by the source?\\n"
            "   - 100: Answer fully supported by source content\\n"
            "   - 80-99: Answer mostly supported, minor gaps\\n"
            "   - 60-79: Answer partially supported, some assumptions\\n"
            "   - 40-59: Answer weakly supported, significant gaps\\n"
            "   - 0-39: Answer not supported by source content\\n"
            "\\n"
            "**B. Question Specificity (0-100 points):**\\n"
            "   - Does the question include sufficient background information?\\n"
            "   - Are conditions clearly specified when answers vary by conditions?\\n"
            "   - Examples of good specificity:\\n"
            "     ✅ '30歳男性がちゃんと応える医療保険EVERに加入した場合、入院給付金はいくらもらえますか？'\\n"
            "     ✅ 'アフラックのがん保険フォルテの保険料支払い方法にはどのような選択肢がありますか？'\\n"
            "   - Examples of poor specificity:\\n"
            "     ❌ '保険金はいくらもらえますか？'\\n"
            "     ❌ '保険料の支払い方法は？'\\n"
            "   - 100: Question is highly specific with all relevant conditions\\n"
            "   - 80-99: Question is mostly specific, minor conditions missing\\n"
            "   - 60-79: Question is moderately specific, some conditions unclear\\n"
            "   - 40-59: Question is somewhat vague, lacks important conditions\\n"
            "   - 0-39: Question is too generic, multiple interpretations possible\\n"
            "\\n"
            "**C. Condition Clarity (0-100 points):**\\n"
            "   - When the answer varies by age, gender, health status, product type, etc., are these conditions clearly stated in the question?\\n"
            "   - Does the question avoid ambiguity that could lead to different answers?\\n"
            "   - Are product names included when answers differ by insurance product?\\n"
            "   - 100: All relevant conditions clearly specified\\n"
            "   - 80-99: Most conditions specified, minor omissions\\n"
            "   - 60-79: Some conditions specified, notable gaps\\n"
            "   - 40-59: Few conditions specified, significant ambiguity\\n"
            "   - 0-39: Conditions not specified, highly ambiguous\\n"
            "\\n"
            "5. Calculate overall_score as weighted average: (source_coverage_score * 0.4 + question_specificity_score * 0.4 + condition_clarity_score * 0.2)\\n"
            "6. Determine overall_rating based on overall_score:\\n"
            "   - 90-100: excellent\\n"
            "   - 70-89: good\\n"
            "   - 50-69: fair\\n"
            "   - 0-49: poor\\n"
            "7. Provide specific feedback:\\n"
            "   - strengths: 2-3 positive aspects of the Q&A\\n"
            "   - improvement_areas: 2-3 areas that need improvement\\n"
            "   - specific_suggestions: 2-3 concrete suggestions for improvement\\n"
            "8. Set needs_improvement = true if overall_score < 80\\n"
            "9. Set improvement_priority: 'high' if score < 50, 'medium' if 50-79, 'low' if 80+\\n"
            "10. All text fields must be in Japanese.\\n"
            "11. Return exactly ONE QAEvaluation object with all required fields."
        ),
        output_type=QAEvaluation,
        model=config["model"],
    )

    try:
        result = await Runner.run(evaluation_agent, input=f"Evaluate Q&A quality: {basic_qa.question}")
        return result.final_output if result.final_output else None
    except Exception as e:
        print(f"    ⚠️ Q&A評価エラー: {e}")
        return None

# --- エージェント6: Q&A改善専用 ---
async def improve_qa_based_on_feedback(
    basic_qa: BasicQAPair,
    evaluation: QAEvaluation,
    source_identifier: str,
    text_content: str,
    model_name: str
) -> Optional[BasicQAPair]:
    """
    評価フィードバックに基づいてQ&Aを改善する専用エージェント
    """
    # 個別エージェント設定を取得
    config = agent_config.get_agent_config("qa_improvement")
    
    improvement_agent = Agent(
        name="QA Improvement Specialist",
        instructions=(
            "You are a specialized Q&A improvement assistant focused on enhancing insurance-related Q&A pairs based on evaluation feedback.\\n"
            f"1. Source context: {source_identifier}\\n"
            f"2. Source text content: \\n---SOURCE CONTENT BEGIN---\\n{text_content}\\n---SOURCE CONTENT END---\\n"
            f"3. Original Q&A to improve:\\n"
            f"   Question: {basic_qa.question}\\n"
            f"   Answer: {basic_qa.answer}\\n"
            "4. Evaluation feedback received:\\n"
            f"   - Overall Score: {evaluation.overall_score}/100 ({evaluation.overall_rating})\\n"
            f"   - Source Coverage: {evaluation.source_coverage_score}/100\\n"
            f"   - Question Specificity: {evaluation.question_specificity_score}/100\\n"
            f"   - Condition Clarity: {evaluation.condition_clarity_score}/100\\n"
            f"   - Strengths: {', '.join(evaluation.strengths)}\\n"
            f"   - Improvement Areas: {', '.join(evaluation.improvement_areas)}\\n"
            f"   - Specific Suggestions: {', '.join(evaluation.specific_suggestions)}\\n"
            "\\n"
            "5. Based on the evaluation feedback, create an improved version of the Q&A pair:\\n"
            "\\n"
            "**For Question Improvement:**\\n"
            "   - Add specific conditions (age, gender, health status, product names) when missing\\n"
            "   - Make the question more specific and less ambiguous\\n"
            "   - Include relevant background information\\n"
            "   - Ensure the question clearly indicates the scope of the answer\\n"
            "\\n"
            "**For Answer Improvement:**\\n"
            "   - Ensure all information is directly supported by the source content\\n"
            "   - Remove any unsupported claims or assumptions\\n"
            "   - Add relevant details from the source when appropriate\\n"
            "   - Make the answer more comprehensive while staying factual\\n"
            "   - Address the specific conditions mentioned in the improved question\\n"
            "\\n"
            "6. Focus on addressing the specific improvement areas identified in the evaluation\\n"
            "7. The improved question and answer MUST be in Japanese\\n"
            "8. Ensure the improved Q&A addresses all the concerns raised in the evaluation\\n"
            f"9. The source_url must be exactly: '{source_identifier}'\\n"
            "10. Return exactly ONE BasicQAPair object with the improved question and answer"
        ),
        output_type=BasicQAPair,
        model=config["model"],
    )

    try:
        result = await Runner.run(improvement_agent, input=f"Improve Q&A based on feedback: {basic_qa.question}")
        
        if result.final_output:
            improved_qa = result.final_output
            # source_urlの修正
            if improved_qa.source_url != source_identifier:
                qa_dict = improved_qa.model_dump()
                qa_dict["source_url"] = source_identifier
                return BasicQAPair(**qa_dict)
            return improved_qa
    except Exception as e:
        print(f"    ⚠️ Q&A改善エラー: {e}")
    
    return None

# --- 統合関数: 評価・改善サイクル付きQ&A生成 ---
async def generate_complete_qa_with_evaluation(
    source_identifier: str,
    text_content: str,
    existing_qa_for_source_display: List[str],
    model_name: str,
    attempt_number: int,
    max_improvement_iterations: int = 2
) -> Optional[QAPair]:
    """
    評価・改善サイクル付きで完全なQ&Aペアを生成
    """
    # Step 1: 基本Q&A生成
    basic_qa = await generate_basic_qa(
        source_identifier,
        text_content,
        existing_qa_for_source_display,
        model_name,
        attempt_number
    )
    
    if not basic_qa:
        print(f"    ❌ 基本Q&A生成失敗")
        return None
    
    print(f"    ✅ 基本Q&A生成成功: {basic_qa.question[:50]}...")
    
    # Step 2: Q&A品質評価
    print(f"    🔍 Q&A品質評価中...")
    evaluation = await evaluate_qa_quality(
        basic_qa,
        source_identifier,
        text_content,
        model_name
    )
    
    if not evaluation:
        print(f"    ⚠️ 評価失敗、基本Q&Aで続行")
        current_qa = basic_qa
        evaluation_score = None
        evaluation_rating = None
        improvement_iterations = 0
    else:
        print(f"    📊 評価完了: {evaluation.overall_score}/100 ({evaluation.overall_rating})")
        print(f"    📈 内訳: ソース整合性={evaluation.source_coverage_score}, 質問特定性={evaluation.question_specificity_score}, 条件明確性={evaluation.condition_clarity_score}")
        
        current_qa = basic_qa
        evaluation_score = evaluation.overall_score
        evaluation_rating = evaluation.overall_rating
        improvement_iterations = 0
        
        # Step 3: 改善が必要な場合は改善サイクル実行
        if evaluation.needs_improvement and evaluation.improvement_priority in ["high", "medium"]:
            print(f"    🔧 改善必要 (優先度: {evaluation.improvement_priority})")
            
            for iteration in range(max_improvement_iterations):
                print(f"    🔄 改善試行 {iteration + 1}/{max_improvement_iterations}")
                
                improved_qa = await improve_qa_based_on_feedback(
                    current_qa,
                    evaluation,
                    source_identifier,
                    text_content,
                    model_name
                )
                
                if improved_qa:
                    print(f"    ✅ Q&A改善成功")
                    print(f"    📝 改善前: {current_qa.question[:40]}...")
                    print(f"    📝 改善後: {improved_qa.question[:40]}...")
                    
                    # 改善版を再評価
                    re_evaluation = await evaluate_qa_quality(
                        improved_qa,
                        source_identifier,
                        text_content,
                        model_name
                    )
                    
                    if re_evaluation and re_evaluation.overall_score > evaluation.overall_score:
                        print(f"    📈 品質向上: {evaluation.overall_score} → {re_evaluation.overall_score}")
                        current_qa = improved_qa
                        evaluation = re_evaluation
                        evaluation_score = re_evaluation.overall_score
                        evaluation_rating = re_evaluation.overall_rating
                        improvement_iterations = iteration + 1
                        
                        # 十分な品質に達した場合は改善サイクル終了
                        if re_evaluation.overall_score >= 80:
                            print(f"    🎯 目標品質達成 ({re_evaluation.overall_score}/100)")
                            break
                    else:
                        print(f"    📊 改善効果限定的、元のQ&Aを採用")
                        break
                else:
                    print(f"    ❌ Q&A改善失敗")
                    break
                
                await asyncio.sleep(1)  # 改善サイクル間の待機
        else:
            print(f"    ✅ 品質良好、改善不要")
    
    # Step 4-6: メタデータ生成（既存の3エージェント並列実行）
    print(f"    🔍 メタデータ分析中...")
    
    persona_task = generate_persona(current_qa, source_identifier, text_content, model_name)
    category_task = generate_category(current_qa, source_identifier, text_content, model_name)  
    keywords_task = generate_keywords(current_qa, source_identifier, text_content, model_name)
    
    persona_result, category_result, keywords_result = await asyncio.gather(
        persona_task, category_task, keywords_task, return_exceptions=True
    )
    
    # メタデータ結果の処理
    persona = "一般的な情報収集者"
    if isinstance(persona_result, PersonaResult):
        persona = persona_result.questioner_persona
        print(f"    ✅ ペルソナ分析成功: {persona}")
    else:
        print(f"    ⚠️ ペルソナ分析失敗、デフォルト値使用: {persona}")
    
    category = "その他"
    if isinstance(category_result, CategoryResult):
        category = category_result.information_category
        print(f"    ✅ カテゴリ分類成功: {category}")
    else:
        print(f"    ⚠️ カテゴリ分類失敗、デフォルト値使用: {category}")
    
    keywords = ["保険", "情報"]
    if isinstance(keywords_result, KeywordsResult):
        keywords = keywords_result.related_keywords
        print(f"    ✅ キーワード抽出成功: {keywords}")
    else:
        print(f"    ⚠️ キーワード抽出失敗、デフォルト値使用: {keywords}")
    
    # Step 7: 完全なQ&Aペアを構築
    complete_qa = QAPair(
        question=current_qa.question,
        answer=current_qa.answer,
        source_url=current_qa.source_url,
        questioner_persona=persona,
        information_category=category,
        related_keywords=keywords,
        evaluation_score=evaluation_score,
        evaluation_rating=evaluation_rating,
        improvement_iterations=improvement_iterations
    )
    
    return complete_qa

# --- 統合関数: 基本Q&A生成 + 3つのメタデータエージェント（評価なし） ---
async def generate_complete_qa_without_evaluation(
    source_identifier: str,
    text_content: str,
    existing_qa_for_source_display: List[str],
    model_name: str,
    attempt_number: int
) -> Optional[QAPair]:
    """
    完全なQ&Aペア（基本Q&A + 3つのメタデータエージェント）を生成（評価・改善なし）
    """
    # Step 1: 基本Q&A生成
    basic_qa = await generate_basic_qa(
        source_identifier,
        text_content,
        existing_qa_for_source_display,
        model_name,
        attempt_number
    )
    
    if not basic_qa:
        print(f"    ❌ 基本Q&A生成失敗")
        return None
    
    print(f"    ✅ 基本Q&A生成成功: {basic_qa.question[:50]}...")
    
    # Step 2-4: 3つのメタデータエージェントを並列実行
    print(f"    🔍 メタデータ分析中...")
    
    # 並列でメタデータ生成
    persona_task = generate_persona(basic_qa, source_identifier, text_content, model_name)
    category_task = generate_category(basic_qa, source_identifier, text_content, model_name)
    keywords_task = generate_keywords(basic_qa, source_identifier, text_content, model_name)
    
    # 並列実行
    persona_result, category_result, keywords_result = await asyncio.gather(
        persona_task, category_task, keywords_task, return_exceptions=True
    )
    
    # 結果の検証とデフォルト値設定
    persona = "一般的な情報収集者"  # デフォルト
    if isinstance(persona_result, PersonaResult):
        persona = persona_result.questioner_persona
        print(f"    ✅ ペルソナ分析成功: {persona}")
    else:
        print(f"    ⚠️ ペルソナ分析失敗、デフォルト値使用: {persona}")
    
    category = "その他"  # デフォルト
    if isinstance(category_result, CategoryResult):
        category = category_result.information_category
        print(f"    ✅ カテゴリ分類成功: {category}")
    else:
        print(f"    ⚠️ カテゴリ分類失敗、デフォルト値使用: {category}")
    
    keywords = ["保険", "情報"]  # デフォルト
    if isinstance(keywords_result, KeywordsResult):
        keywords = keywords_result.related_keywords
        print(f"    ✅ キーワード抽出成功: {keywords}")
    else:
        print(f"    ⚠️ キーワード抽出失敗、デフォルト値使用: {keywords}")
    
    # Step 5: 完全なQ&Aペアを構築（評価なしモード）
    complete_qa = QAPair(
        question=basic_qa.question,
        answer=basic_qa.answer,
        source_url=basic_qa.source_url,
        questioner_persona=persona,
        information_category=category,
        related_keywords=keywords,
        evaluation_score=None,  # 評価なしモード
        evaluation_rating=None,
        improvement_iterations=0  # 改善実行なし
    )
    
    return complete_qa

# --- 並列処理対応: ファイルI/O ロック管理 ---
import threading
import time
from datetime import datetime

# ファイル書き込み用ロック
file_lock = threading.Lock()

def collect_existing_qa_for_source(source_identifier: str, outfile: str) -> List[str]:
    """
    指定されたソースIDに関する既存Q&Aを収集
    """
    existing_qa_display = []
    if os.path.exists(outfile):
        try:
            with file_lock:  # ファイル読み込み時もロック
                with jsonlines.open(outfile, "r") as reader:
                    for qa_obj_dict in reader:
                        if qa_obj_dict.get("source_url") == source_identifier:
                            q = qa_obj_dict.get("question")
                            a = qa_obj_dict.get("answer")
                            if q and a:
                                existing_qa_display.append(f"- Q: {q}\\n  A: {a}")
        except Exception as e:
            print(f"警告: 既存Q&A収集中にエラー ({source_identifier}): {e}")
    return existing_qa_display

def save_qa_to_file(qa: QAPair, outfile: str) -> bool:
    """
    Q&Aをファイルに安全に保存
    """
    try:
        with file_lock:  # ファイル書き込み時のロック
            with jsonlines.open(outfile, "a") as writer:
                writer.write(qa.model_dump())
        return True
    except Exception as e:
        print(f"Q&A保存エラー: {e}")
        return False

async def process_single_entry(
    entry_data: Tuple[int, Dict[str, Any]],
    outfile: str,
    model_name: str,
    source_id_field: str,
    content_field: str,
    max_q_per_entry: int,
    global_existing_qa_set: Set[Tuple[str, str]],
    enable_evaluation: bool = True,
    max_improvement_iterations: int = 2
) -> int:
    """
    単一エントリの処理（評価・改善サイクル付き）
    """
    i, entry = entry_data
    
    source_identifier = entry.get(source_id_field)
    text_content = entry.get(content_field)
    
    if not source_identifier:
        print(f"⚠️ エントリ {i+1}: '{source_id_field}' が見つからないか空です。スキップします。")
        return 0
    if not text_content:
        print(f"⚠️ エントリ {i+1}: '{content_field}' が見つからないか空です。スキップします。")
        return 0

    print(f"🔄 エントリ {i+1} を処理中: {source_identifier}")

    # このソースの既存Q&Aを収集
    existing_qa_for_current_source_display = collect_existing_qa_for_source(source_identifier, outfile)
    
    # エントリ内でのQ&A生成は逐次実行（品質重視）
    current_entry_added_count = 0
    for attempt in range(max_q_per_entry):
        print(f"  📝 エントリ {i+1}, 試行 {attempt + 1}/{max_q_per_entry}")
        
        if enable_evaluation:
            # 評価・改善サイクル付きでQ&A生成
            complete_qa = await generate_complete_qa_with_evaluation(
                source_identifier,
                text_content,
                existing_qa_for_current_source_display,
                model_name,
                attempt + 1,
                max_improvement_iterations
            )
        else:
            # 従来の4段階処理（評価なし）
            complete_qa = await generate_complete_qa_without_evaluation(
                source_identifier,
                text_content,
                existing_qa_for_current_source_display,
                model_name,
                attempt + 1
            )
        
        if complete_qa:
            current_qa_tuple = (complete_qa.question, complete_qa.answer)
            
            # グローバル重複チェック（スレッドセーフ）
            with file_lock:
                is_duplicate = current_qa_tuple in global_existing_qa_set
                if not is_duplicate:
                    global_existing_qa_set.add(current_qa_tuple)
            
            if not is_duplicate:
                # ファイルに保存
                if save_qa_to_file(complete_qa, outfile):
                    # 次の試行で使用するため、このエントリの既存リストに追加
                    existing_qa_for_current_source_display.append(
                        f"- Q: {complete_qa.question}\\n  A: {complete_qa.answer}"
                    )
                    current_entry_added_count += 1
                    print(f"    ✅ 完全Q&A生成成功")
                else:
                    print(f"    ❌ Q&A保存失敗")
            else:
                print(f"    ⚠️ 重複のためスキップ: {complete_qa.question[:50]}...")
        else:
            print(f"    ❌ Q&A生成失敗")
        
        # 評価・改善サイクルがある場合は待機時間を延長
        wait_time = 5 if enable_evaluation else 3
        await asyncio.sleep(wait_time)
    
    if current_entry_added_count > 0:
        print(f"✨ エントリ {i+1}: {current_entry_added_count} 件を新たに生成")
    else:
        print(f"ℹ️ エントリ {i+1}: 新しいQ&Aは生成されませんでした")
    
    # エントリ間の待機
    await asyncio.sleep(0.5)
    return current_entry_added_count

# --- エントリレベル並列処理のメイン関数 ---
async def process_jsonl_parallel_entries(
    input_jsonl_path: str,
    outfile: str,
    model_name: str,
    source_id_field: str,
    content_field: str,
    max_q_per_entry: int = 1,
    max_entries_to_process: int = -1,
    max_concurrent_entries: int = 1,  # 評価・改善サイクルのため1に変更
    enable_evaluation: bool = True,
    max_improvement_iterations: int = 2
):
    """
    エントリレベル並列処理でJSONLファイルを処理（評価・改善サイクル付き）
    """
    if not os.path.exists(input_jsonl_path):
        print(f"❌ エラー: 入力ファイル '{input_jsonl_path}' が見つかりません。")
        return

    # 既存Q&Aの読み込み
    global_existing_qa_set: Set[Tuple[str, str]] = set()
    if os.path.exists(outfile):
        try:
            with jsonlines.open(outfile, "r") as reader:
                for qa_obj_dict in reader:
                    global_existing_qa_set.add((qa_obj_dict.get("question"), qa_obj_dict.get("answer")))
            print(f"📂 既存の出力ファイル '{outfile}' から {len(global_existing_qa_set)} 件のQ&Aを読み込みました。")
        except Exception as e:
            print(f"⚠️ 警告: 既存の出力ファイル '{outfile}' の読み込み中にエラー: {e}")

    # エントリを読み込み
    entries = []
    with jsonlines.open(input_jsonl_path, "r") as reader:
        for i, entry in enumerate(reader):
            if max_entries_to_process != -1 and i >= max_entries_to_process:
                break
            entries.append((i, entry))

    processing_mode = "評価・改善サイクル付き" if enable_evaluation else "標準4段階"
    agent_count = "6個 (Q&A生成 + 評価 + 改善 + ペルソナ + カテゴリ + キーワード)" if enable_evaluation else "4個 (Q&A生成 + ペルソナ + カテゴリ + キーワード)"
    
    print(f"\n🚀 Q&A生成システム開始")
    print(f"=" * 60)
    print(f"📂 入力ファイル: {input_jsonl_path}")
    print(f"💾 出力ファイル: {outfile}")
    print(f"🔢 処理エントリ数: {len(entries)}")
    print(f"🤖 使用モデル: {model_name}")
    print(f"📊 エントリあたりQ&A数: {max_q_per_entry}")
    print(f"⚡ 最大並列数: {max_concurrent_entries}")
    print(f"🔧 処理モード: {processing_mode}")
    print(f"🤖 使用エージェント数: {agent_count}")
    if enable_evaluation:
        print(f"🔄 最大改善試行回数: {max_improvement_iterations}")
    print(f"=" * 60)
    
    start_time = time.time()

    # 並列処理用セマフォ
    semaphore = asyncio.Semaphore(max_concurrent_entries)
    
    async def process_entry_with_semaphore(entry_data):
        async with semaphore:
            return await process_single_entry(
                entry_data,
                outfile,
                model_name,
                source_id_field,
                content_field,
                max_q_per_entry,
                global_existing_qa_set,
                enable_evaluation,
                max_improvement_iterations
            )
    
    # 並列実行
    tasks = [process_entry_with_semaphore(entry_data) for entry_data in entries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 結果集計
    total_newly_added = sum(r for r in results if isinstance(r, int))
    error_count = sum(1 for r in results if isinstance(r, Exception))
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n📊 処理完了サマリー")
    print(f"=" * 60)
    print(f"🎉 新規Q&A生成数: {total_newly_added} 件")
    print(f"⏱️ 処理時間: {processing_time:.2f} 秒")
    print(f"⚡ 平均処理速度: {len(entries) / processing_time:.2f} エントリ/秒")
    if error_count > 0:
        print(f"❌ エラー発生エントリ数: {error_count} 件")
    print(f"💾 出力ファイル: {outfile}")
    print(f"🔧 処理モード: {processing_mode}")
    print(f"=" * 60)

# --- 下位互換性のためのエイリアス ---
async def process_jsonl_single_qa_mode(
    input_jsonl_path: str,
    outfile: str,
    model_name: str,
    source_id_field: str,
    content_field: str,
    max_q_per_entry: int = 3,
    max_entries_to_process: int = -1,
):
    """
    下位互換性のための関数（並列処理版を呼び出し）
    """
    await process_jsonl_parallel_entries(
        input_jsonl_path,
        outfile,
        model_name,
        source_id_field,
        content_field,
        max_q_per_entry,
        max_entries_to_process,
        max_concurrent_entries=1  # 逐次処理モード
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="6段階エージェント処理でJSONLファイルからQ&Aペア生成（評価・改善サイクル付き）")
    parser.add_argument("--input_jsonl", required=True, help="入力JSONLファイルパス")
    parser.add_argument("--outfile", required=True, help="出力JSONLファイルパス")
    parser.add_argument("--model", default="gpt-4o-mini", help="使用するモデル名（デフォルト設定上書き用）")
    parser.add_argument("--source_id_field", default="url", help="ソースID用フィールド名")
    parser.add_argument("--content_field", default="response_body", help="コンテンツ用フィールド名")
    parser.add_argument("--max_q_per_entry", type=int, default=1, help="エントリあたり最大Q&A数")
    parser.add_argument("--max_entries", type=int, default=-1, help="処理するエントリ数上限（-1で全て）")
    parser.add_argument("--max_concurrent", type=int, default=1, help="並列実行数")
    parser.add_argument("--disable_evaluation", action="store_true", help="評価・改善サイクルを無効化")
    parser.add_argument("--max_improvement_iterations", type=int, default=2, help="改善サイクル最大回数")
    
    # エージェント設定オプション
    parser.add_argument("--quality_mode", choices=["standard", "balanced", "all_premium"], default="balanced", 
                        help="品質モード: standard（全てgpt-4o-mini）, balanced（重要タスクgpt-4o）, all_premium（全てgpt-4o）")
    parser.add_argument("--qa_generation_model", help="Q&A生成エージェント専用モデル")
    parser.add_argument("--evaluation_model", help="評価エージェント専用モデル")
    parser.add_argument("--improvement_model", help="改善エージェント専用モデル")
    parser.add_argument("--persona_model", help="ペルソナ分析エージェント専用モデル")
    parser.add_argument("--category_model", help="カテゴリ分析エージェント専用モデル")
    parser.add_argument("--keywords_model", help="キーワード抽出エージェント専用モデル")
    
    args = parser.parse_args()
    
    # エージェント設定の適用
    agent_config.set_quality_mode(args.quality_mode)
    
    # 個別モデル設定の適用
    if args.qa_generation_model:
        agent_config.set_agent_model("qa_generation", args.qa_generation_model)
    if args.evaluation_model:
        agent_config.set_agent_model("qa_evaluation", args.evaluation_model)
    if args.improvement_model:
        agent_config.set_agent_model("qa_improvement", args.improvement_model)
    if args.persona_model:
        agent_config.set_agent_model("persona_analysis", args.persona_model)
    if args.category_model:
        agent_config.set_agent_model("category_analysis", args.category_model)
    if args.keywords_model:
        agent_config.set_agent_model("keyword_extraction", args.keywords_model)
    
    # 設定表示
    print("🚀 6段階エージェント処理開始")
    print(f"📄 入力ファイル: {args.input_jsonl}")
    print(f"📋 出力ファイル: {args.outfile}")
    print(f"🎯 品質モード: {args.quality_mode}")
    print(f"🔄 評価・改善: {'有効' if not args.disable_evaluation else '無効'}")
    agent_config.print_config()
    print()
    
    asyncio.run(process_jsonl_parallel_entries(
        args.input_jsonl,
        args.outfile,
        args.model,
        args.source_id_field,
        args.content_field,
        args.max_q_per_entry,
        args.max_entries,
        args.max_concurrent,
        not args.disable_evaluation,  # enable_evaluation
        args.max_improvement_iterations
    ))

"""
🔧 【エージェント個別設定の使用例】

# 1. 品質モード設定
--quality_mode standard     # 全エージェントgpt-4o-mini（低コスト）
--quality_mode balanced     # 重要タスクのみgpt-4o（推奨）
--quality_mode all_premium  # 全エージェントgpt-4o（最高品質）

# 2. 個別エージェント設定
--qa_generation_model gpt-4o        # Q&A生成のみ高性能モデル
--evaluation_model gpt-4o-mini      # 評価は効率重視
--improvement_model gpt-4o          # 改善は高性能モデル
--persona_model gpt-4o-mini         # ペルソナ分析は効率重視
--category_model gpt-4o-mini        # カテゴリ分類は効率重視
--keywords_model gpt-4o-mini        # キーワード抽出は効率重視

# 3. コスト最適化の例（重要な処理のみ高性能モデル）
python Create_QA_from_jsonl_alt_fixed.py \
    --input_jsonl data.jsonl \
    --outfile output.jsonl \
    --quality_mode standard \
    --qa_generation_model gpt-4o \
    --evaluation_model gpt-4o

# 4. 最高品質設定の例
python Create_QA_from_jsonl_alt_fixed.py \
    --input_jsonl data.jsonl \
    --outfile output.jsonl \
    --quality_mode all_premium

# 5. カスタム設定の例
python Create_QA_from_jsonl_alt_fixed.py \
    --input_jsonl data.jsonl \
    --outfile output.jsonl \
    --qa_generation_model gpt-4o \
    --improvement_model gpt-4o \
    --evaluation_model gpt-4o-mini \
    --persona_model gpt-4o-mini \
    --category_model gpt-4o-mini \
    --keywords_model gpt-4o-mini

これにより、以下のような柔軟な運用が可能です：
✅ コスト重視: 分類系タスクは効率的なモデル、生成系は高性能モデル
✅ 品質重視: 全てのエージェントで最高性能モデルを使用
✅ バランス重視: 重要度に応じてモデルを使い分け
✅ 実験的運用: 特定のエージェントのみ異なるモデルでテスト

評価・改善サイクル付き6段階エージェント処理でJSONLファイルからQ&Aペア生成

このスクリプトは、JSONLファイルから高品質なQ&Aペアを生成するために、
6つの専門エージェントを使用します：

1. Q&A生成エージェント: 基本的な質問-回答ペアを生成
2. Q&A評価エージェント: 生成されたQ&Aの品質を評価
3. Q&A改善エージェント: 評価フィードバックに基づいてQ&Aを改善
4. ペルソナ分析エージェント: 質問者のペルソナを特定
5. カテゴリ分類エージェント: 情報カテゴリを分類
6. キーワード抽出エージェント: 関連キーワードを抽出

各エージェントは特定の専門分野に集中し、高品質な結果を提供します。
評価・改善サイクルにより、自動的にQ&Aの品質向上を図ります。
"""
