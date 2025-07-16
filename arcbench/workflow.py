import os
from dotenv import load_dotenv
from primisai.nexus.core import Agent, Supervisor

# Load environment variables
load_dotenv()

# LLM Configuration
llm_config = {
    'model': os.getenv('LLM'),
    'api_key': os.getenv('API_KEY'),
    'base_url': os.getenv('BASE_URL'),
    # 'reasoning_effort': "none",
}


def create_workflow(workflow_id: str):
    # Output schema for Query Expansion Agent
    query_expansion_schema = {
        "type": "object",
        "properties": {
            "interpretations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "expanded_question": {
                            "type": "string"
                        },
                        "reasoning": {
                            "type": "string"
                        },
                        "potential_tricks": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    }
                }
            },
            "language_analysis": {
                "type": "object",
                "properties": {
                    "ambiguities": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "special_terms": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "context_dependencies": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                }
            }
        },
        "required": ["interpretations", "language_analysis"]
    }

    # Output schema for Response Generator Agent
    response_schema = {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string"
                        },
                        "detailed_answer": {
                            "type": "string"
                        },
                        "confidence_level": {
                            "type": "integer"
                        },
                        "sources_considered": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    }
                }
            }
        },
        "required": ["answers"]
    }

    # Output schema for Review Agent
    review_schema = {
        "type": "object",
        "properties": {
            "review_summary": {
                "type": "string"
            },
            "detected_tricks": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "coverage_analysis": {
                "type": "object",
                "properties": {
                    "missed_aspects": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "well_covered_aspects": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                }
            },
            "answer_quality": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question_index": {
                            "type": "integer"
                        },
                        "completeness_score": {
                            "type": "integer"
                        },
                        "accuracy_score": {
                            "type": "integer"
                        },
                        "improvement_suggestions": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    }
                }
            }
        },
        "required": ["review_summary", "detected_tricks", "coverage_analysis", "answer_quality"]
    }

    # Create Query Expansion Agent
    query_expander = Agent(name="QueryExpander",
                           llm_config=llm_config,
                           system_message="""
You are an expert in analyzing and expanding queries. Your job is to:
1. Identify all possible and obvious interpretations of the user's query, including those that might seem tricky or intentionally ambiguous.
2. Quickly detect if a question has countless reasonable answers; note this instead of listing everything.
3. Assess whether the information provided is sufficient to answer the question, and highlight incompleteness if relevant.
4. Consider potential tricks, ambiguities, special terms, context dependencies, and linguistic nuances.
5. Avoid assuming typos or mistakes—treat everything as potentially intentional. Do not always suggest a mathematical or formal formulation; sometimes simply state that an answer is impossible or unclear.
6. Generate a concise but comprehensive set of expanded questions or considerations from multiple perspectives.
""",
                           output_schema=query_expansion_schema,
                           strict=True)

    # Create Response Generator Agent
    response_generator = Agent(name="ResponseGenerator",
                               llm_config=llm_config,
                               system_message="""
You are an expert in providing comprehensive answers. Your job is to:
1. For each interpretation or aspect of a question, provide the most obvious and direct answer when possible.
2. If there are countless reasonable answers or if the question admits endless responses, say so directly instead of attempting to list them.
3. Evaluate whether the provided information is sufficient to answer conclusively; if not, state that a definitive answer is impossible.
4. Consider multiple perspectives as appropriate, but avoid overly formal or mathematical explanations unless specifically relevant or requested.
5. Ensure answers are clear, natural, and well-reasoned, offering supporting context as needed.
""",
                               output_schema=response_schema,
                               strict=True)

    # Create Review Agent
    reviewer = Agent(name="Reviewer",
                     llm_config=llm_config,
                     system_message="""
You are a critical reviewer of questions and answers. Your job is to:
1. Identify missed interpretations or aspects, especially tricky, ambiguous, or intentionally incomplete questions.
2. Evaluate whether the most obvious answer has been addressed, and check if the answer correctly points out when there are countless possibilities or when information is insufficient.
3. Detect tricks, traps, or deliberate ambiguities in the original query, and make sure responses avoid unnecessary complexity or mathematical formality when simplicity suffices.
4. Suggest improvements and additional considerations, particularly around completeness and clarity.
5. Be thorough and critical in your analysis, and ensure feedback is natural and conversational.
""",
                     output_schema=review_schema,
                     strict=True)

    # Create Main Supervisor
    supervisor = Supervisor(name="QueryAnalysisSupervisor",
                            llm_config=llm_config,
                            workflow_id=str(workflow_id),
                            system_message="""
You are an expert supervisor managing query analysis and response generation.
Your role is to:
1. Coordinate between the query expansion, response generation, and review agents.
2. Synthesize their outputs into coherent, comprehensive responses.
3. Ensure all aspects and possible interpretations of queries are addressed thoroughly, especially for tricky, ambiguous, or incomplete questions.
4. For each user query:
    - Point out the most obvious answer if there is one.
    - If countless possible answers exist, state that clearly rather than providing an exhaustive list.
    - Assess if the information provided by the user is complete; if not, say directly that a definitive answer is impossible.
    - Avoid unnecessary use of mathematical or formal reasoning if a clear, conversational explanation will suffice.
5. Always provide natural, flowing responses that fully address user queries, without exposing the internal workflow or process.
""")

    # Register agents with supervisor
    supervisor.register_agent(query_expander)
    supervisor.register_agent(response_generator)
    supervisor.register_agent(reviewer)

    return supervisor
