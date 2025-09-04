from pydantic_settings import BaseSettings


class AWSConfig(BaseSettings):
    """AWS cloud configuration"""

    region: str
    access_key_id: str
    access_key_secret: str
    bucket_name: str

    class Config:
        alias_generator = lambda x: f"aws_{x}".upper()
        populate_by_name = True
        extra = "ignore"


class OSSConfig(BaseSettings):
    """OSS cloud configuration"""

    access_key_id: str
    access_key_secret: str
    end_point: str
    bucket_name: str

    class Config:
        alias_generator = lambda x: f"oss_{x}".upper()
        populate_by_name = True
        extra = "ignore"
