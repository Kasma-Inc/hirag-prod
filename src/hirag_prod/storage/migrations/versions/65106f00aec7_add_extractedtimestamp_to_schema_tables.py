"""Add extractedTimestamp to schema tables

Revision ID: 65106f00aec7
Revises: 
Create Date: 2025-10-03 08:08:39.437071

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '65106f00aec7'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def column_exists(table_name, column_name):
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = inspector.get_columns(table_name)
    return any(c['name'] == column_name for c in columns)

def upgrade() -> None:
    """Add extractedTimestamp column to schema tables if it doesn't exist."""
    tables = ['Chunks', 'Files', 'Items', 'Nodes', 'Graph', 'Triplets']
    for table in tables:
        if not column_exists(table, 'extractedTimestamp'):
            op.add_column(table, sa.Column('extractedTimestamp', sa.DateTime(), nullable=True))

def downgrade() -> None:
    """Remove extractedTimestamp column from schema tables if it exists."""
    tables = ['Chunks', 'Files', 'Items', 'Nodes', 'Graph', 'Triplets']
    # Reverse the order for downgrade to match potential dependency order
    tables.reverse()
    for table in tables:
        if column_exists(table, 'extractedTimestamp'):
            op.drop_column(table, 'extractedTimestamp')