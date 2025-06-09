# 金融术语RAG系统 / Finterm

## 简介
这是一个关于金融术语的项目，用于提交极客时间训练营的一项作业。它的主要功能包括：
- milvus数据库管理 ![milvus数据库管理.jpeg](images/milvus%E6%95%B0%E6%8D%AE%E5%BA%93%E7%AE%A1%E7%90%86.jpeg)
- 金融术语NER ![拼写错误检查.jpeg](images/%E6%8B%BC%E5%86%99%E9%94%99%E8%AF%AF%E6%A3%80%E6%9F%A5.jpeg)
- 金融术语标准化 ![金融术语NER.jpeg](images/%E9%87%91%E8%9E%8D%E6%9C%AF%E8%AF%ADNER.jpeg)
- 拼写错误检查 ![金融术语标准化.jpeg](images/%E9%87%91%E8%9E%8D%E6%9C%AF%E8%AF%AD%E6%A0%87%E5%87%86%E5%8C%96.jpeg)

## 安装步骤
1. 克隆仓库: `git clone https://github.com/PFNOW/Finterm.git`
2. 安装依赖: `pip install -r requirements.txt`

## 使用方法
1. 进入项目目录: `cd ./src`
2. 启动项目: ` python gradio_server.py`
3. 访问项目: `http://localhost:7860`

## 贡献指南
如果您想为这个项目做出贡献，请遵循以下步骤：
1. Fork 仓库
2. 创建新的分支: `git checkout -b feature/新的功能`
3. 提交您的更改: `git commit -m '添加新的功能'`
4. 推送到分支: `git push origin feature/新的功能`
5. 提交 Pull Request

## 许可证
这个项目采用 Apache 2.0 许可证。详情请查看 LICENSE 文件。
