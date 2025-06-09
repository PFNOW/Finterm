from milvus_database import Milvus
from logger import LOG


class StdService:
    """
    标准化服务类，用于标准化金融词汇
    """

    def __init__(self, config):
        """
        初始化标准化服务

        :param config: 配置对象，包含所有的模型配置参数。
        """
        self.milvus_connection = Milvus(config)

    def standardization_service(self, database_name: str, collection_name: str, entities: list[dict], limit: int = 5):
        """
        标准化服务，将实体列表中的实体进行标准化处理

        :param database_name: 数据库路径
        :param collection_name: 集合名称
        :param entities: 实体列表，每个实体是字典结构，包含实体名称和实体类型
        :param limit: 返回结果的数量
        :return: 标准化后的实体列表
        """
        results = []
        for entity in entities:
            query = entity['word']
            # 获取查询的向量表示
            LOG.info(f"Searching for similar terms to: {query}")
            search_result = self.milvus_connection.search(database_name, collection_name, query, limit)

            for hit in search_result[0]:
                results.append({
                    "query": query,
                    "concept": hit['entity'].get('concept'),
                    "concept_group": hit['entity'].get('concept_group'),
                    "distance": float(hit['distance'])
                })
        LOG.info(f"Search result: {results}")
        return results

if __name__ == "__main__":
    # 测试代码
    from config import Config
    config = Config()
    std_service = StdService(config)
    std_service.standardization_service(
        database_name="finterm_bge_m3.db",
        collection_name="finance_rag",
        entities=[
            {'entity': 'B-ORG', 'word': 'Apple Inc.', 'start': 0, 'end': 10, 'score': 0.8515142798423767},
            {'entity': 'B-LOC', 'word': 'New York City', 'start': 56, 'end': 69, 'score': 0.9949582815170288}
        ],
        limit=4
    )


