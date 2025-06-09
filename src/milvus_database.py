from pymilvus import model
import os # 导入os库用于文件操作
import pandas as pd
from tqdm import tqdm
from logger import LOG
from dotenv import load_dotenv
load_dotenv()
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

class Milvus:
    def __init__(self, config):
        """
        初始化 LLM 类，根据配置选择使用的模型（OpenAI 或 huggingface）。

        :param config: 配置对象，包含所有的模型配置参数。
        TODO: 后续根据需要，将embedding模型独立出来
        """
        self.Milvus_database_path = "./db/"
        self.config = config
        self.embedding_type = config.embedding_type.lower()  # 获取模型类型并转换为小写
        if self.embedding_type == "openai":
            self.embedding_function = model.dense.OpenAIEmbeddingFunction(
                model_name=config.embedding_model_name if config.embedding_model_name else'text-embedding-3-large'
            )  # 创建OpenAI客户端实例
        elif self.embedding_type == "huggingface":
            self.embedding_function = model.dense.SentenceTransformerEmbeddingFunction(
                model_name=config.embedding_model_name,
                device=config.embedding_device,
                trust_remote_code=True
            )
        else:
            LOG.error(f"不支持的模型类型: {self.embedding_type}")
            raise ValueError(f"不支持的模型类型: {self.embedding_type}")  # 如果模型类型不支持，抛出错误

    def get_milvus_database(self):
        """
        获取所有 Milvus 数据库的名称。

        :return: 数据库名称列表。
        """
        File_list = os.listdir(self.Milvus_database_path)
        Milvus_database_list = [s for s in File_list if s.endswith(".db")]
        LOG.debug(f"Milvus数据库列表: {Milvus_database_list}")
        return Milvus_database_list

    def delete_milvus_database(self, database_name):
        """
        删除指定的 Milvus 数据库。

        :param database_name: Milvus 数据库名称。
        :return: 删除结果。
        """
        if not database_name:
            LOG.info("请选择数据库")
            return "请选择数据库"
        db_path = self.Milvus_database_path + database_name
        if not os.path.exists(db_path):
            return f"数据库{database_name}不存在"
        try:
            os.remove(db_path)
            LOG.info(f"Milvus数据库 {database_name} 删除成功")
            return f"Milvus数据库 {database_name} 删除成功"
        except FileNotFoundError:
            LOG.error(f"Milvus数据库 {database_name} 不存在")
            return f"Milvus数据库 {database_name} 不存在"
        except Exception as e:
            LOG.error(f"删除文件时发生错误: {e}")
            return f"删除文件时发生错误: {e}"

    def get_milvus_collection(self, database_name):
        """
        列出指定数据库中的所有集合。

        :param database_name: Milvus 数据库名称。
        :return: 集合名称列表。
        """
        if not database_name:
            LOG.info("请选择数据库")
            return "请选择数据库"
        db_path = self.Milvus_database_path + database_name
        client = MilvusClient(db_path)
        collection_list = client.list_collections()
        LOG.debug(f"Milvus集合列表: {collection_list}")
        return collection_list

    def drop_collection(self, database_name: str, collection_name: str):
        """
        删除指定的 Milvus 集合。

        :param database_name: Milvus 数据库名称。
        :param collection_name: 集合名称。
        :return: None
        """
        if not database_name or not collection_name:
            LOG.info("请选择完善信息")
            return "请选择完善信息"
        # 连接到 Milvus
        db_path = self.Milvus_database_path + database_name
        client = MilvusClient(db_path)
        client.drop_collection(collection_name)
        LOG.info(f"成功删除集合 {collection_name}")
        return f"成功删除集合 {collection_name}"

    def create_new_collection(self, file_path: str, database_name: str, collection_name: str, description: str = "") -> str:
        """
        创建新的 Milvus 集合，并加载 CSV 文件中的数据。

        :param file_path: CSV 文件路径。
        :param database_name: Milvus 数据库路径。
        :param collection_name: 集合名称。
        :param description: 集合描述。
        :return: None
        """
        if not database_name or not collection_name:
            LOG.info("请选择完善信息")
            return "请选择完善信息"

        # 连接到 Milvus
        db_path = self.Milvus_database_path + database_name
        client = MilvusClient(db_path)

        # 加载数据
        LOG.info(f"加载CSV文件{file_path}中的数据")
        df = pd.read_csv(file_path,
                         dtype=str,
                         low_memory=False,
                         names=["concept", "concept_group"],
                         ).fillna("NA")

        # 获取向量维度
        sample_doc = "Sample Text"
        sample_embedding = self.embedding_function([sample_doc])[0]
        vector_dim = len(sample_embedding)

        # 构造Schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="concept", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="concept_group", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="input_file", dtype=DataType.VARCHAR, max_length=500),
        ]
        schema = CollectionSchema(fields,
                                  description=description,
                                  enable_dynamic_field=True)

        # 如果集合不存在，创建集合
        if not client.has_collection(collection_name):
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
            )
            LOG.info(f"Created new collection: {collection_name}")

        # # 在创建集合后添加索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",  # 指定要为哪个字段创建索引，这里是向量字段
            index_type="AUTOINDEX",  # 使用自动索引类型，Milvus会根据数据特性选择最佳索引
            metric_type="COSINE",  # 使用余弦相似度作为向量相似度度量方式
            params={"nlist": 1024}  # 索引参数：nlist表示聚类中心的数量，值越大检索精度越高但速度越慢
        )

        client.create_index(
            collection_name=collection_name,
            index_params=index_params
        )

        # 批量处理
        batch_size = 1024

        for start_idx in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            # 准备文档
            docs = []
            for _, row in batch_df.iterrows():
                doc_parts = [row['concept']]
                docs.append(" ".join(doc_parts))

            # 生成嵌入
            try:
                embeddings = self.embedding_function(docs)
                LOG.info(f"生成第{start_idx // batch_size + 1}批向量")
            except Exception as e:
                LOG.error(f"生成第{start_idx // batch_size + 1}批向量时出错: {e}")
                continue

            # 准备数据
            data = [
                {
                    "vector": embeddings[idx],
                    "concept": str(row['concept']),
                    "concept_group": str(row['concept_group']),
                    "input_file": file_path
                } for idx, (_, row) in enumerate(batch_df.iterrows())
            ]

            # 插入数据
            try:
                res = client.insert(
                    collection_name=collection_name,
                    data=data
                )
                LOG.info(f"Inserted batch {start_idx // batch_size + 1}, result: {res}")
            except Exception as e:
                LOG.error(f"Error inserting batch {start_idx // batch_size + 1}: {e}")

        LOG.info("成功创建集合.")
        return f"成功创建集合 {collection_name}"



    def search(self, database_name: str, collection_name: str, query: str, limit: int = 5) -> list[list[dict]]:
        """
        在 Milvus 集合中搜索与查询最相似的文档。

        :param database_name: Milvus 数据库。
        :param collection_name: 集合名称。
        :param query: 查询字符串。
        :param limit: 返回结果的数量。
        :return: 与查询最相似的文档列表。
        """

        # 连接到 Milvus
        db_path = self.Milvus_database_path + database_name
        client = MilvusClient(db_path)

        # 示例查询
        query_embeddings = self.embedding_function([query])

        # 搜索余弦相似度最高的
        search_result = client.search(
            collection_name=collection_name,
            data=[query_embeddings[0].tolist()],
            limit=limit,
            output_fields=["concept",  "concept_group"]
        )
        client.release_collection(collection_name)
        LOG.info(f"Search result for {query}: {search_result}")
        return search_result

    def insert(self, database_name: str, collection_name: str, data: dict | list[dict], partition_name:str = "") -> dict:
        """
        将数据插入到 Milvus 集合中。

        :param database_name: Milvus 数据库路径。
        :param collection_name: 集合名称。
        :param data: 要插入的数据项。
        :param partition_name: 分区名称。
        :return: 插入结果。
        """
        # 连接到 Milvus
        db_path = self.Milvus_database_path + database_name
        client = MilvusClient(db_path)

        # 插入数据
        insert_result = client.insert(collection_name = collection_name, data = data, partition_name = partition_name)
        client.release_collection(collection_name)
        LOG.info(f"Inserted item: {insert_result}")
        return insert_result

    def delete(self, database_name: str, collection_name: str, ids: list | str | int | None, filter:str = "", partition_name:str = "") -> dict:
        """
        从 Milvus 集合中删除指定 ID 的数据。

        :param database_name: Milvus 数据库路径。
        :param collection_name: 集合名称。
        :param ids: 要删除的数据项的 ID 列表。
        :param filter: 过滤条件。
        :param partition_name: 分区名称。
        :return: 删除结果。
        """
        # 连接到 Milvus
        db_path = self.Milvus_database_path + database_name
        client = MilvusClient(db_path)

        # 删除数据
        delete_result = client.delete(collection_name=collection_name, ids=ids, filter=filter, partition_tag=partition_name)
        client.release_collection(collection_name)
        LOG.info(f"Deleted result: {delete_result}")
        return delete_result

    def update(self, database_name: str, collection_name: str, data: dict | list[dict], filter: str = "", partition_name: str = "") -> dict:
        """
        更新 Milvus 集合中的数据。

        :param database_name: Milvus 数据库路径。
        :param collection_name: 集合名称。
        :param data: 要更新的数据项。
        :param filter: 过滤条件。
        :param partition_name: 分区名称。
        :return: 更新结果。
        """
        # 连接到 Milvus
        db_path = self.Milvus_database_path + database_name
        client = MilvusClient(db_path)

        # 更新数据
        update_result = client.upsert(collection_name=collection_name, data=data, filter=filter, partition_tag=partition_name)
        client.release_collection(collection_name)
        LOG.info(f"Updated result: {update_result}")
        return update_result



if __name__ == "__main__":
    from config import Config
    config = Config()
    m = Milvus(config)
    # m.create_new_collection("data/万条金融标准术语.csv", "finterm_bge_m3", "finance_rag", "finance concepts")
    # m.search("db/snomed_bge_m3.db", "finance_rag", "A-B")
    # m.get_milvus_collection("finterm_bge_m3.db")
    m.get_milvus_database()
