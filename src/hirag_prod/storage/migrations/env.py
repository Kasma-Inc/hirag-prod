import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

from hirag_prod.configs.functions import get_config_manager, initialize_config_manager
from hirag_prod.resources.functions import initialize_resource_manager

# Import project modules
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root / "src"))

try:
    from hirag_prod.schema import Base

    target_metadata = Base.metadata
except ImportError as e:
    print(f"Error importing project modules: {e}")
    target_metadata = None

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config


# Get DB URL from environment variable
def get_database_url():
    """Get database URL for Alembic migrations"""
    db_url = None

    # Try config manager first
    try:

        async def db_url_setup():
            initialize_config_manager({"debug": False}, None)
            await initialize_resource_manager()

        asyncio.run(db_url_setup())
        config_manager = get_config_manager()
        if config_manager and hasattr(config_manager, "postgres_url_sync"):
            db_url = config_manager.postgres_url_sync
            print("Retrieved database URL from config manager")
    except Exception as e:
        print(f"Error retrieving database URL via get_config_manager(): {e}")
        db_url = None

    # Fallback to environment variables if config manager failed
    if not db_url:
        env_settings = os.getenv("ENV", "dev")
        env_var_name = (
            "POSTGRES_URL_NO_SSL"
            if env_settings == "prod"
            else "POSTGRES_URL_NO_SSL_DEV"
        )
        db_url = os.getenv(env_var_name)
        if db_url:
            print(f"Retrieved database URL from environment variable {env_var_name}")
        else:
            print(f"Environment variable {env_var_name} not found")

    if db_url:
        # Ensure proper sync driver format for Alembic
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        elif db_url.startswith("postgresql+asyncpg://"):
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)

        # Mask credentials properly - they are between npg_ and @
        masked_url = db_url
        if "npg_" in db_url and "@" in db_url:
            start = db_url.find("npg_")
            end = db_url.find("@", start)
            if start != -1 and end != -1:
                masked_url = db_url[:start] + "***" + db_url[end:]
        print(f"Using database URL: {masked_url}")
        return db_url

    return None


# Set database URL
try:
    db_url = get_database_url()
    if not db_url:
        raise ValueError(
            "Database URL is not set. Please set POSTGRES_URL_NO_SSL_DEV or POSTGRES_URL_NO_SSL environment variable."
        )
    config.set_main_option("sqlalchemy.url", db_url)
except Exception as e:
    print(f"Error setting database URL: {e}")
    raise


# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
