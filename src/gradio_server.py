import gradio as gr  # 导入gradio库用于创建GUI

from config import Config  # 导入配置管理模块
from milvus_database import Milvus  # 导入Milvus数据库模块
from corr_service import CorrectionService  # 导入修改模块
from std_service import StdService  # 导入标准化服务模块
from ner_service import NERService  # 导入命名实体识别服务模块
from llm import LLM  # 导入可能用于处理语言模型的LLM类
from logger import LOG  # 导入日志记录器

# 创建各个组件的实例
config = Config()
Milvus = Milvus(config)
StdService = StdService(config)
NERService = NERService()

# 定义一个回调函数，用于根据 Radio 组件的选择返回不同的 Dropdown 选项
def update_model_list(model_type):
    if model_type == "openai":
        return gr.Dropdown(choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], label="选择模型")
    elif model_type == "ollama":
        return gr.Dropdown(choices=["llama3.1", "qwen3:14b"], label="选择模型")
    else:
        return None

def delete_milvus_database(database_list):
    LOG.info("开始删除数据库")
    db_management = Milvus.delete_milvus_database(database_list)
    database_list = gr.Dropdown(choices=Milvus.get_milvus_database(), label="Milvus数据库")
    return database_list, db_management

def show_collections(database_list):
    LOG.info("开始显示集合")
    collections = Milvus.get_milvus_collection(database_list)
    collection_list = gr.Dropdown(choices=collections, label="集合列表")
    return collection_list

def drop_collection(database_name, collection_list):
    LOG.info("开始删除集合")
    db_management = Milvus.drop_collection(database_name, collection_list)
    collection_list = gr.Dropdown(label = "集合列表")
    return collection_list, db_management

def import_data(database_name, collection_name, csv_file, description):
    LOG.info("开始导入数据")
    db_management = Milvus.create_new_collection(csv_file, database_name, collection_name, description)
    return db_management

def ner_service(input_text, options):
    # 高亮处理
    LOG.info("开始命名实体识别")
    results = NERService.process(input_text, options)
    named_entities = results["entities"]
    highlighted_text = [(named_entity["word"], named_entity["entity"]) for named_entity in named_entities]
    matched_sorted = sorted(highlighted_text, key=lambda x: input_text.find(x[0]))
    # 初始化结果列表和指针
    result = []
    last_end = 0
    for entity, label in matched_sorted:
        start = input_text.find(entity, last_end)
        if start == -1:
            continue  # 如果没有找到，跳过（根据题目描述应该不会发生）
        # 添加未匹配的前面部分
        unmatched = input_text[last_end:start]
        if unmatched:
            result.append((unmatched, None))
        # 添加匹配的实体
        result.append((entity, label))
        last_end = start + len(entity)
    # 添加剩余的未匹配部分
    unmatched = input_text[last_end:]
    if unmatched:
        result.append((unmatched, None))
    return result

def std_service(database_name, collection_name, input_text, limit):
    LOG.info("开始标准化")
    options = ["combine_person", "combine_location", "combine_organization", "combine_miscellaneous"]
    results = NERService.process(input_text, options)
    named_entities = results["entities"]
    std_result = StdService.standardization_service(database_name, collection_name, named_entities, limit)
    return std_result

def corr_service(model_type, model_name, corr_text):
    LOG.info("开始纠错")
    if model_type:
        config.llm_model_type = model_type
    if model_name:
        if model_type == "openai":
            config.openai_model_name = model_name
        else:
            config.ollama_model_name = model_name
    llm = LLM(config)  # 创建语言模型实例
    correctionService = CorrectionService(llm, "Correction")
    result = correctionService.correct(corr_text)
    return result

# 创建 Gradio 界面
with gr.Blocks(title="FinTerm") as demo:

    with gr.Tab("Milvus数据库管理"):
        gr.Markdown("# Milvus数据库管理")
        databases = Milvus.get_milvus_database()
        with gr.Group():
            database_list= gr.Dropdown(choices = databases, label = "Milvus数据库", value="")
            delete_database_button = gr.Button("删除数据库")
            collection_list = gr.Dropdown(label = "集合列表")
            database_list.change(fn=show_collections, inputs=database_list, outputs=collection_list)
            delete_collection_button = gr.Button("删除集合")
            db_management = gr.TextArea(label = "操作结果",interactive = False)
            delete_database_button.click(fn=delete_milvus_database, inputs=database_list, outputs=[database_list, db_management])
            delete_collection_button.click(fn=drop_collection, inputs=[database_list, collection_list], outputs=[collection_list, db_management])
        with gr.Group():
            database_name = gr.Textbox(label="数据库名称")
            collection_name = gr.Textbox(label="集合名称")
            csv_file = gr.File(label="上传CSV文件", type="filepath", file_types=[".csv"])
            description = gr.TextArea(label="关于集合的描述", info="选填", lines=7)
            import_data_button = gr.Button("导入数据")
            import_data_result = gr.TextArea(label="导入结果", interactive=False)
            import_data_button.click(fn=import_data, inputs=[database_name, collection_name, csv_file, description], outputs=import_data_result)

    with gr.Tab("命名实体识别"):
        with gr.Column(scale=1):
            gr.Markdown("# 命名实体识别")
            ner_text = gr.TextArea(label="输入文本", lines=7)
            options_checkbox = gr.CheckboxGroup(choices=[("人名", "combine_person"),("地名", "combine_location"), ("企业名称", "combine_organization"), ("产品、事件等其他金融实体", "combine_miscellaneous")], label="选择高亮内容",
                                                info="识别选项")
            output_textbox = gr.HighlightedText(
                label="Diff",
                combine_adjacent=True,
                show_inline_category = True,
                show_legend=False,
                color_map={"LOC": "blue", "PER": "red", "ORG": "green", "MISC": "yellow"},
            ),
            submit_button = gr.Button("提交")
            submit_button.click(ner_service, inputs=[ner_text, options_checkbox], outputs=output_textbox)

    with gr.Tab("命名标准化"):
        gr.Markdown("# 命名标准化")
        database_list = gr.Dropdown(choices=databases, label="Milvus数据库", value="")
        collection_list = gr.Dropdown(label="集合列表")
        database_list.change(fn=show_collections, inputs=database_list, outputs=collection_list)
        std_text = gr.TextArea(label="输入文本", lines=7)
        limit = gr.Number(label = "查询条数", info = "可选", value = 5, minimum = 1, precision = 0)
        std_button = gr.Button("提交")
        std_result = gr.JSON(label="操作结果", open=True)
        std_button.click(fn = std_service, inputs=[database_list, collection_list, std_text, limit], outputs=std_result)

    with gr.Tab("拼写纠错"):
        gr.Markdown("# 拼写纠错")
        model_type = gr.Radio(["openai", "ollama"], label="模型类型", info="使用 OpenAI GPT API 或 Ollama 私有化模型服务", value = "")
        model_name = gr.Dropdown(choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], label="选择模型", value = "", interactive= True)
        model_type.change(fn=update_model_list, inputs=model_type, outputs=model_name)
        spell_text = gr.TextArea(label="输入文本", lines=7)
        correction_button = gr.Button("提交")
        correction_text = gr.TextArea(label="修改后文本", interactive=False)
        correction_button.click(fn=corr_service, inputs=[model_type, model_name, spell_text], outputs=correction_text)


if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
    # 可选带有用户认证的启动方式, 启动界面并设置为公共可访问
    # demo.launch(share=True, server_name="0.0.0.0", auth=("user", "1234"))