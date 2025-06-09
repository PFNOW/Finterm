from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from logger import LOG

class NERService:
    """
    金融命名实体识别服务
    """
    def __init__(self):
        # 指定模型缓存目录
        self.cache_dir = './cache'

        # 加载预训练的金融领域命名实体识别模型和分词器
        tokenizer = AutoTokenizer.from_pretrained("shrimp1106/bert-finetuned-finance_ner", cache_dir=self.cache_dir)
        model = AutoModelForTokenClassification.from_pretrained("shrimp1106/bert-finetuned-finance_ner", cache_dir=self.cache_dir)
        self.pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, device="cpu")

    def process(self, text:str, options:list) -> dict:
        """
        金融命名实体识别

        :param text: 待识别的文本
        :param options: 命名实体识别选项
        :return: 实体识别结果
        """
        # 进行NER
        LOG.info("开始进行实体识别")
        ner_results = self.pipeline(text)
        LOG.info(f"实体识别结果: {ner_results}")

        # 确保结果是实体列表
        if isinstance(ner_results, dict):
            ner_results = ner_results.get('entities', [])

        # 合并相关实体
        LOG.info("开始合并实体")
        combined_result = self._combine_entities(ner_results, text, options)

        # 返回结果
        return {
            "text": text,
            "entities": combined_result
        }

    def _combine_entities(self, entities, text, options):
        """
        合并相关实体

        :param entities: 实体列表
        :param text: 待识别的文本
        :param options: 合并选项
        :return: 合并后的实体列表
        """
        # 处理合并选项
        labels = []
        if 'combine_person' in options:
            labels += ['B-PER', 'I-PER']
        if 'combine_location' in options:
            labels += ['B-LOC', 'I-LOC']
        if 'combine_organization' in options:
            labels += ['B-ORG', 'I-ORG']
        if 'combine_miscellaneous' in options:
            labels += ['B-MISC', 'I-MISC']

        # 合并相关实体的结果
        combined_result = []
        current_entities = []
        for entity in entities:
            entity["score"] = float(entity["score"])
            if entity['entity'] in labels:
                if entity["entity"].startswith("B-"):
                    if current_entities:
                        combined_entity = self._try_combine(current_entities, text)
                        combined_result.append(combined_entity)
                        current_entities = []
                    current_entities.append(entity)
                else:
                    current_entities.append(entity)

        if current_entities:
            combined_entity = self._try_combine(current_entities, text)
            combined_result.append(combined_entity)
        LOG.info(f"合并后的实体: {combined_result}")
        return combined_result

    def _try_combine(self, current_entities, text):
        """
        尝试将当前实体合并
        """
        # 合并当前实体
        start = current_entities[0]['start']
        end = current_entities[-1]['end']
        combined_entity = {
            'entity': current_entities[0]['entity'][2:],
            'word':  text[start: end],
            'start': start,
            'end': end,
            'score': sum([entity['score'] for entity in current_entities]) / len(current_entities)
        }
        return combined_entity

"""
O (Outside)表示当前词不属于任何命名实体（普通词）。
B-PER (Begin-Person)表示人名实体的起始词，如 "Barack" 在 "Barack Obama" 中。
I-PER (Inside-Person)表示人名实体的后续词，如 "Obama" 在 "Barack Obama" 中。
B-LOC (Begin-Location)表示地点实体的起始词，如 "New" 在 "New York" 中。
I-LOC (Inside-Location)表示地点实体的后续词，如 "York" 在 "New York" 中。
B-ORG (Begin-Organization)表示组织实体的起始词，如 "United" 在 "United Nations" 中。
I-ORG (Inside-Organization)表示组织实体的后续词，如 "Nations" 在 "United Nations" 中。
B-MISC (Begin-Miscellaneous)表示其他杂项实体的起始词（如事件、产品等），如 "World" 在 "World War II" 中。
I-MISC (Inside-Miscellaneous)表示杂项实体的后续词，如 "War" 和 "II" 在 "World War II" 中。
"""

if __name__ == '__main__':
    ner_service = NERService()
    results = ner_service.process('Apple Inc. achieved a revenue of $50 billion in 2022 in New York City.', ["combine_organization", "combine_location"])
    named_entities = results["entities"]
    highlighted_text = [(named_entity["word"], named_entity["entity"]) for named_entity in named_entities]

    text = "Apple Inc. achieved a revenue of $50 billion in 2022 in New York City."

    matched_sorted = sorted(highlighted_text, key=lambda x: text.find(x[0]))

    # 初始化结果列表和指针
    result = []
    last_end = 0

    for entity, label in matched_sorted:
        start = text.find(entity, last_end)
        if start == -1:
            continue  # 如果没有找到，跳过（根据题目描述应该不会发生）

        # 添加未匹配的前面部分
        unmatched = text[last_end:start]
        if unmatched:
            result.append((unmatched, None))

        # 添加匹配的实体
        result.append((entity, label))
        last_end = start + len(entity)

    # 添加剩余的未匹配部分
    unmatched = text[last_end:]
    if unmatched:
        result.append((unmatched, None))

    print(result)
