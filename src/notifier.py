import smtplib
import markdown2
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from logger import LOG

class Notifier:
    def __init__(self, email_settings):
        self.email_settings = email_settings
    
    def notify(self, repo, report):
        """
        发送 GitHub 项目报告邮件
        :param repo: 仓库名称
        :param report: 报告内容
        """
        if self.email_settings:
            subject = f"[GitHub] {repo} 进展简报"
            self._send_email(subject, report)
        else:
            LOG.warning("邮件设置未配置正确，无法发送 GitHub 报告通知")
    
    def _send_email(self, subject, report):
        LOG.info(f"准备发送邮件:{subject}")
        msg = MIMEMultipart()
        msg['From'] = self.email_settings['from']
        msg['To'] = self.email_settings['to']
        msg['Subject'] = subject
        
        # 将Markdown内容转换为HTML
        html_report = markdown2.markdown(report)

        msg.attach(MIMEText(html_report, 'html'))
        try:
            with smtplib.SMTP_SSL(self.email_settings['smtp_server'], self.email_settings['smtp_port']) as server:
                LOG.debug("登录SMTP服务器")
                server.login(msg['From'], self.email_settings['password'])
                server.sendmail(msg['From'], msg['To'], msg.as_string())
                LOG.info("邮件发送成功！")
        except Exception as e:
            LOG.error(f"发送邮件失败：{str(e)}")

if __name__ == '__main__':
    from config import Config
    config = Config()
    notifier = Notifier(config.email)

    # 测试 GitHub 报告邮件通知
    test_repo = "项目进展"
    test_report = """
# 项目进展

## 时间周期：2024-08-24

## 项目进度
已完成专业词汇标准化
"""
    notifier.notify(test_repo, test_report)

