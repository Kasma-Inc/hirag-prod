import asyncio
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import nltk
from googletrans import Translator
from redis.asyncio import ConnectionPool, Redis
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from hirag_prod._llm import (
    ChatCompletion,
    EmbeddingService,
    LocalChatService,
    LocalEmbeddingService,
    create_chat_service,
    create_embedding_service,
)
from hirag_prod._utils import log_error_info
from hirag_prod.configs.functions import (
    get_config_manager,
    get_envs,
    get_hi_rag_config,
)
from hirag_prod.reranker import Reranker, create_reranker
from hirag_prod.resources.functions import timing_logger


class ResourceManager:
    _instance: Optional["ResourceManager"] = None  # Per-process instances
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "ResourceManager":
        """Ensure singleton pattern - one instance per process."""
        if cls._instance is None:
            with cls._lock:  # Move lock outside to prevent race condition
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, resource_dict: Optional[Dict] = None):
        """Initialize the resource manager (called only once per process)."""
        # Check if this specific instance has been initialized
        if getattr(self, "_created", False):
            return

        # Database resources
        self._db_engine: Optional[AsyncEngine] = (
            resource_dict.get("db_engine", None) if resource_dict else None
        )
        self._session_maker: Optional[async_sessionmaker] = (
            resource_dict.get("session_maker", None) if resource_dict else None
        )

        # Redis resources
        self._redis_pool: Optional[ConnectionPool] = (
            resource_dict.get("redis_pool", None) if resource_dict else None
        )

        # Services
        self._chat_service: Optional[Union[ChatCompletion, LocalChatService]] = None
        self._embedding_service: Optional[
            Union[EmbeddingService, LocalEmbeddingService]
        ] = None

        # Translator
        self._translator: Optional[Translator] = (
            resource_dict.get("translator", None) if resource_dict else None
        )

        # Reranker
        self._reranker: Optional[Reranker] = (
            resource_dict.get("reranker", None) if resource_dict else None
        )

        # Cleanup operation list
        self._cleanup_operation_list: List[Tuple[str, Any]] = []

        # Initialization lock (will be created when first accessed)
        self._init_lock: Optional[asyncio.Lock] = None
        self._lock_creation_lock: threading.Lock = (
            threading.Lock()
        )  # Thread-safe lock creation

        # Mark this instance as created
        self._created: bool = True

        # This instance is not initialized currently
        self._initialized: bool = False

        logging.info(f"🚀 ResourceManager instance created for process {os.getpid()}")

    def _ensure_init_lock(self) -> asyncio.Lock:
        """Ensure initialization lock exists in a thread-safe manner."""
        if self._init_lock is None:
            with self._lock_creation_lock:
                if self._init_lock is None:
                    self._init_lock = asyncio.Lock()
        return self._init_lock

    @timing_logger("ResourceManager.initialize")
    async def initialize(
        self,
    ) -> None:
        """
        Initialize all resources. This method should be called only once
        during application startup in the main process.
        """
        async with self._ensure_init_lock():
            if self._initialized:
                logging.warning("⚠️ ResourceManager already initialized, skipping...")
                return

            logging.info(f"🔄 Initializing ResourceManager...")

            try:
                # Download nltk data
                nltk.download("wordnet")

                # Initialize database engine with connection pool
                if (not self._db_engine) or (not self._session_maker):
                    await self._initialize_database()

                # Initialize Redis connection pool
                if not self._redis_pool:
                    await self._initialize_redis()

                # Initialize services
                if not self._chat_service:
                    self._chat_service = create_chat_service()
                if not self._embedding_service:
                    self._embedding_service = create_embedding_service(
                        default_batch_size=get_hi_rag_config().embedding_batch_size
                    )

                # Initialize Translator
                if not self._translator:
                    self._translator = Translator()

                # Initialize Reranker
                if not self._reranker:
                    self._reranker = create_reranker()

                # Reverse the cleanup operation list to ensure proper cleanup
                self._cleanup_operation_list.reverse()

                # Mark this instance as initialized
                self._initialized = True

                logging.info(
                    f"✅ ResourceManager initialization completed successfully"
                )

            except Exception as e:
                log_error_info(
                    logging.ERROR, f"❌ Failed to initialize ResourceManager", e
                )
                # Only cleanup resources that were actually initialized to avoid cleanup errors
                await self.cleanup(ensure_init_lock=False)
                raise

    @timing_logger("Database initialization")
    async def _initialize_database(self) -> None:
        """Initialize database engine with optimized connection pool settings."""

        async def _cleanup():
            if self._db_engine:
                try:
                    await asyncio.wait_for(self._db_engine.dispose(), timeout=5.0)
                finally:
                    self._db_engine = None
                    self._session_maker = None

        self._cleanup_operation_list.append(("database", _cleanup))

        logging.info(f"🗄 Initializing database engine...")

        postgres_url: str = get_config_manager().postgres_url_async

        logging.info(f"Using database URL: {postgres_url.split('@')[0]}@***")

        # Create async engine with connection pool

        self._db_engine = create_async_engine(
            postgres_url,
            # Connection pool settings for high performance
            pool_size=20,  # Number of persistent connections
            max_overflow=20,  # Additional connections beyond pool_size
            pool_timeout=30,  # Timeout for getting connection from pool
            pool_recycle=3600,  # Recycle connections every hour
            pool_pre_ping=True,  # Verify connections before use
            echo=False,  # Set to True for SQL logging in development
        )

        # Create session maker
        self._session_maker = async_sessionmaker(
            self._db_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logging.info(f"✅ Database engine initialized successfully")

    @timing_logger("Redis connection pool initialization")
    async def _initialize_redis(self) -> None:
        """Initialize Redis connection pool."""

        async def _cleanup():
            self._redis_pool = None

        self._cleanup_operation_list.append(("redis", _cleanup))

        logging.info(f"🔗 Initializing Redis connection pool...")

        redis_url: str = get_envs().REDIS_URL

        logging.info(
            f"Using Redis URL: {redis_url.split('@')[0] if '@' in redis_url else redis_url.split('://')[0] + '://***'}"
        )

        # Create connection pool with optimized settings
        self._redis_pool = ConnectionPool.from_url(
            redis_url,
            max_connections=20,  # Maximum connections in pool
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30,
        )

        # Test the connection
        redis_client = Redis(connection_pool=self._redis_pool)
        try:
            await redis_client.ping()
            logging.info(f"✅ Redis connection pool initialized successfully")
        except Exception as e:
            log_error_info(
                logging.ERROR, f"❌ Failed to connect to Redis", e, raise_error=True
            )
            raise
        finally:
            await redis_client.aclose()

    def is_initialized(self) -> bool:
        """Determine whether this instance is initialized."""
        return self._initialized

    def get_session_maker(self) -> async_sessionmaker:
        """Get the database session maker."""
        if self._session_maker is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._session_maker

    def get_db_engine(self) -> AsyncEngine:
        """Get the database engine."""
        if self._db_engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._db_engine

    def get_redis_client(self, **kwargs: Any) -> Redis:
        """Get a Redis client from the connection pool."""
        if self._redis_pool is None:
            raise RuntimeError("Redis not initialized. Call initialize() first.")
        return Redis(connection_pool=self._redis_pool, **kwargs)

    def get_chat_service(self) -> Union[ChatCompletion, LocalChatService]:
        """Get the chat service instance."""
        if self._chat_service is None:
            raise RuntimeError("Chat service not initialized. Call initialize() first.")
        return self._chat_service

    def get_embedding_service(self) -> Union[EmbeddingService, LocalEmbeddingService]:
        """Get the embedding service instance."""
        if self._embedding_service is None:
            raise RuntimeError(
                "Embedding service not initialized. Call initialize() first."
            )
        return self._embedding_service

    def get_translator(self) -> Translator:
        """Get the translator instance."""
        if self._translator is None:
            raise RuntimeError("Translator not initialized. Call initialize() first.")
        return self._translator

    def get_reranker(self) -> Reranker:
        """Get the reranker instance."""
        if self._reranker is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")
        return self._reranker

    async def cleanup(self, ensure_init_lock: bool = True) -> None:
        try:
            if ensure_init_lock:
                await self._ensure_init_lock().acquire()
            """Cleanup all resources. Should be called during application shutdown."""
            logging.info(f"🧹 Cleaning up ResourceManager...")

            cleanup_errors = []

            for cleanup_operation in self._cleanup_operation_list:
                resource_name: str = cleanup_operation[0]
                resource_name_first_upper: str = (
                    cleanup_operation[0][0].upper() + cleanup_operation[0][1:]
                )
                try:
                    logging.info(f"🧹 Cleaning up {resource_name}...")
                    await cleanup_operation[1]()
                    logging.info(f"✅ {resource_name_first_upper} cleanup completed")
                except asyncio.TimeoutError as e:
                    log_error_info(
                        logging.WARNING,
                        f"{resource_name_first_upper} cleanup timed out",
                        e,
                    )
                    cleanup_errors.append(
                        f"{resource_name_first_upper} cleanup timeout"
                    )
                except asyncio.CancelledError as e:
                    log_error_info(
                        logging.WARNING,
                        f"{resource_name_first_upper} cleanup was cancelled",
                        e,
                    )
                    cleanup_errors.append(
                        f"{resource_name_first_upper} cleanup cancelled"
                    )
                except Exception as e:
                    log_error_info(
                        logging.ERROR, f"Error cleaning up {resource_name}", e
                    )
                    cleanup_errors.append(f"{resource_name_first_upper}: {e}")

            # Log summary
            if cleanup_errors:
                logging.warning(
                    f"⚠ ResourceManager cleanup completed with {len(cleanup_errors)} errors: {cleanup_errors}"
                )
            else:
                logging.info(f"✅ ResourceManager cleanup completed successfully")
            # Don't raise exceptions during shutdown to prevent blocking application exit

            self._initialized = False
        finally:
            if ensure_init_lock:
                self._ensure_init_lock().release()
