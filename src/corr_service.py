import os
from logger import LOG  # 导入日志模块

class CorrectionService:
    """
    纠正金融报告中的拼写错误。
    """
    def __init__(self, llm, service):
        """
        初始化服务。
        :param llm: LLM 实例，用于校正拼写错误。
        :param service: 需要使用的服务列表。
        """
        self.llm = llm  # 初始化时接受一个LLM实例，用于后续校正报告
        self.service = service
        self.prompts = {}  # 存储提示词模板
        self._preload_prompts()

    def _preload_prompts(self):
        """
        预加载所有可能的提示文件，并存储在字典中。
        """
        prompt_file = f"prompts/{self.service}_{self.llm.model}_prompt.txt"
        if not os.path.exists(prompt_file):
            LOG.error(f"提示文件不存在: {prompt_file}")
            raise FileNotFoundError(f"提示文件未找到: {prompt_file}")
        with open(prompt_file, "r", encoding='utf-8') as file:
            self.prompts[self.service] = file.read()

    def correct(self, text):
        """
        校正报告中错误/遗漏的信息。
        :param text: 报告内容。
        :return: 校正后的报告内容。
        """
        system_prompt = self.prompts.get(self.service)
        result = self.llm.generate(system_prompt, text)
        LOG.info(f"校正结果 {result}")
        return result



if __name__ == '__main__':
    from config import Config  # 导入配置管理类
    from llm import LLM

    config = Config()
    llm = LLM(config)

    service = CorrectionService(llm, "correction")

    result = service.correct("Microsoft Inc. reports a revenue of $200 billion in 2022 of Redmond, Washington.")

    LOG.debug(f"校正结果 {result}")