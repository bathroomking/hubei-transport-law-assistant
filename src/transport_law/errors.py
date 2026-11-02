class AppError(Exception):
    """向 API / CLI 返回稳定错误码，不暴露上游响应或密钥。"""

    def __init__(self, code: str, message: str, status_code: int = 502):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
