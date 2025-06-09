import json
import os

class Config:
    def __init__(self):
        with open('config.json', 'r') as f:
            config = json.load(f)

            # 加载邮箱配置
            self.email = config.get('email', {})
            self.email['password'] = os.getenv('EMAIL_PASSWORD', self.email.get('password', ''))

            # 加载 LLM 相关配置
            llm_config = config.get('llm', {})
            self.llm_model_type = llm_config.get('model_type', 'openai')
            self.openai_model_name = llm_config.get('openai_model_name', 'gpt-4o-mini')
            self.ollama_model_name = llm_config.get('ollama_model_name', 'Qwen3:0.6b')
            self.ollama_api_url = llm_config.get('ollama_api_url', 'http://localhost:11434/api/chat')

            # 加载报告类型配置
            self.services = config.get('services', ["correction"])  # 默认服务类型

            # TODO: Slack 配置
            slack_config = config.get('slack', {})
            self.slack_webhook_url = slack_config.get('webhook_url')

            # 加载embedding 模型配置
            embedding_config = config.get('embedding', {})
            self.embedding_type = embedding_config.get('embedding_type', 'huggingface')
            self.embedding_model_name = embedding_config.get('embedding_model_name', 'BAAI/bge-m3')
            self.embedding_device = embedding_config.get('device', 'cpu') # cuda: 0

