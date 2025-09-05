#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import lancedb
import pyarrow as pa
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from hirag_prod._utils import log_error_info
from hirag_prod.configs.functions import get_hi_rag_config
from hirag_prod.storage.pgvector import PGVector
from hirag_prod.storage.retrieval_strategy_provider import RetrievalStrategyProvider

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
console = Console()


class VDBManager:
    """## Vector Database Utilities
    (now use [lancedb](https://www.lancedb.com/) as the vector database)

    ### Database Path:
    By default, the tool looks for the database at `kb/hirag.db`. You can specify a different
    database path using the `--db-path` parameter:

    `python -m src.hirag_prod.storage.vdb_utils --db-path /path/to/your/database.db --overview`

    ### Command Line Usage:
    - Basic operations (uses default db: `kb/hirag.db`)
        - `python -m src.hirag_prod.storage.vdb_utils --help`
        - `python -m src.hirag_prod.storage.vdb_utils --list-tables`
        - `python -m src.hirag_prod.storage.vdb_utils --overview`

    - Custom database path
        - `python -m src.hirag_prod.storage.vdb_utils --db-path /path/to/custom.db --overview`

    - Export single table (use `--output` for specific file)
        - `python -m src.hirag_prod.storage.vdb_utils --table chunks --output chunks.csv`

    - Export all tables (use `--output-dir` for batch directory)
        - `python -m src.hirag_prod.storage.vdb_utils --export all --output-dir exports`

    - Export with custom database and exclude columns
        - `python -m src.hirag_prod.storage.vdb_utils --db-path /path/to/db --table entities --output entities.csv --exclude-columns vector description`

    - Export with row limit
        - `python -m src.hirag_prod.storage.vdb_utils --table chunks --output chunks_sample.csv --limit 100`

    ### Output Options:
    - `--output`: Specify output file for single table export (used with `--table`)
    - `--output-dir`: Specify directory for batch export (used with `--export all`)
    - `--exclude-columns`: List of columns to exclude from export (default: `vector`)
    - `--limit`: Maximum number of rows to export

    ### Module Usage:
    ```python
    from vdb_utils import VDBManager

    async with VDBManager("path/to/db") as vdb:
        # List tables
        tables = await vdb.list_tables()

        # Get table info
        info = await vdb.get_table_info("chunks")

        # Export data
        await vdb.export_table_to_csv("chunks", "output.csv")
    ```
    """

    def __init__(self, db_path: str = "kb/hirag.db"):
        """
        Initialize VDB Manager

        Args:
            db_path: Path to LanceDB database
        """
        self.db_path = Path(db_path)
        self.db: Optional[lancedb.AsyncConnection] = None
        self._table_cache: Dict[str, Any] = {}

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self) -> None:
        """Establish database connection"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        try:
            self.db = await lancedb.connect_async(str(self.db_path))
            logger.info(f"Connected to LanceDB: {self.db_path}")
        except Exception as e:
            log_error_info(logging.ERROR, "Connection failed", e, raise_error=True)

    async def disconnect(self) -> None:
        """Close database connection"""
        if self.db:
            self._table_cache.clear()
            self.db = None
            logger.info("Disconnected from LanceDB")

    async def list_tables(self) -> List[str]:
        """List all tables in database"""
        try:
            return await self.db.table_names()
        except Exception as e:
            log_error_info(logging.ERROR, "Failed to list tables", e)
            return []

    async def _get_table(self, table_name: str):
        """Get table with caching"""
        if table_name not in self._table_cache:
            self._table_cache[table_name] = await self.db.open_table(table_name)
        return self._table_cache[table_name]

    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive table information

        Args:
            table_name: Name of table to analyze

        Returns:
            Dictionary with table metadata
        """
        try:
            table = await self._get_table(table_name)

            # NOTE: remember handle different schema types
            schema = table.schema

            if hasattr(schema, "names"):
                # PyArrow schema
                columns = schema.names
            elif hasattr(schema, "__iter__"):
                # Iterable schema
                try:
                    columns = [field.name for field in schema]
                except Exception as e:
                    log_error_info(logging.WARNING, "Failed to get table columns", e)
                    arrow_data = await table.to_arrow()
                    columns = arrow_data.column_names
            else:
                try:
                    arrow_data = await table.to_arrow()
                    columns = arrow_data.column_names
                except Exception as e:
                    log_error_info(logging.WARNING, "Failed to get table columns", e)
                    columns = ["id", "page_content", "vector"]

            try:
                count_result = await table.count_rows()
            except Exception as e:
                log_error_info(logging.WARNING, "Failed to get row count", e)
                arrow_data = await table.to_arrow()
                count_result = len(arrow_data)

            try:
                sample = await table.query().select(columns).limit(1).to_list()
            except Exception as e:
                log_error_info(logging.WARNING, "Failed to get sample data", e)
                arrow_data = await table.to_arrow()
                if len(arrow_data) > 0:
                    sample_df = arrow_data.to_pandas().head(1)
                    sample = (
                        [sample_df.to_dict("records")[0]] if len(sample_df) > 0 else []
                    )
                else:
                    sample = []

            return {
                "name": table_name,
                "row_count": count_result,
                "columns": columns,
                "schema": schema,
                "sample_data": sample[0] if sample else None,
                "column_types": {col: "unknown" for col in columns},
            }
        except Exception as e:
            log_error_info(logging.ERROR, f"Failed to get info for {table_name}", e)
            return {"name": table_name, "error": str(e)}

    async def export_table_to_csv(
        self,
        table_name: str,
        output_file: str,
        exclude_columns: Optional[Set[str]] = None,
        limit: Optional[int] = None,
    ) -> bool:
        """
        Export table to CSV with options

        Args:
            table_name: Table to export
            output_file: Output CSV path
            exclude_columns: Columns to exclude (default: {'vector'})
            limit: Maximum rows to export

        Returns:
            Success status
        """
        exclude_columns = exclude_columns or {"vector"}

        try:
            table = await self._get_table(table_name)

            arrow_data = await table.to_arrow()
            df = arrow_data.to_pandas()

            if limit:
                df = df.head(limit)

            columns_to_drop = [col for col in df.columns if col in exclude_columns]
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                console.print(
                    f"[yellow]Excluded columns: {', '.join(columns_to_drop)}[/yellow]"
                )

            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].apply(self._serialize_value)

            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False, encoding="utf-8")

            console.print(f"[green]✅ Exported {len(df)} rows to {output_file}[/green]")
            return True

        except Exception as e:
            log_error_info(logging.ERROR, f"Export failed for {table_name}", e)
            console.print(f"[red]❌ Export failed: {e}[/red]")
            return False

    @staticmethod
    def _serialize_value(value: Any) -> str:
        """Serialize complex values for CSV export"""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _format_value(self, value: Any, max_length: int = 100) -> str:
        """Format value for display"""
        if isinstance(value, (list, pa.Array)) and len(value) > 10:
            return f"<array[{len(value)}]>"

        str_val = str(value)
        # NOTE: if the value is a string, and the length is greater than max_length, then truncate the string
        if len(str_val) > max_length:
            return str_val[:max_length] + "..."
        return str_val

    async def display_table_summary(self, table_name: str) -> None:
        """Display formatted table summary"""
        info = await self.get_table_info(table_name)

        if "error" in info:
            console.print(f"[red]❌ {table_name}: {info['error']}[/red]")
            return

        table = Table(title=f"📊 {table_name}", show_header=True)
        table.add_column("Property", style="cyan", width=20)
        table.add_column("Value", style="green")

        table.add_row("Rows", f"{info['row_count']:,}")
        table.add_row("Columns", str(len(info["columns"])))
        table.add_row(
            "Column Names",
            ", ".join(info["columns"][:5])
            + ("..." if len(info["columns"]) > 5 else ""),
        )

        console.print(table)

        if info.get("sample_data"):
            console.print("\n[bold]Sample Record:[/bold]")
            sample = info["sample_data"]

            for key, value in list(sample.items())[:5]:
                formatted = self._format_value(value)
                console.print(f"  • {key}: {formatted}")

            if len(sample) > 5:
                console.print(f"  ... and {len(sample) - 5} more fields")

    async def display_overview(self) -> None:
        """Display database overview with progress"""
        tables = await self.list_tables()

        if not tables:
            console.print("[yellow]⚠️  No tables found in database[/yellow]")
            return

        console.print(
            Panel.fit(
                f"[bold cyan]LanceDB Multimodal Database[/bold cyan]\n"
                f"📍 Path: {self.db_path}\n"
                f"📊 Tables: {len(tables)}",
                border_style="blue",
            )
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing tables...", total=len(tables))

            for table_name in tables:
                progress.update(task, description=f"Analyzing {table_name}...")
                console.print(f"\n[bold]{table_name}[/bold]")
                await self.display_table_summary(table_name)
                progress.advance(task)


async def get_chunk_info(
    chunk_ids: list[str],
    knowledge_base_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> Optional[list[Dict[str, Any]]]:
    """Get a list of chunk records with its headers from vector database by its ids.

    Note: the chunk identifier is stored in the `document_key` column of the
    `chunks` table for LanceDB, and `documentKey` column for PostgreSQL.

    Args:
        chunk_ids: list of chunk ids to get info for
        knowledge_base_id: The id of the knowledge base that the chunk is from (optional)
        workspace_id: The id of the workspace that the chunk is from (optional)

    Returns:
        A list of dicts of the chunk rows with its headers if found, otherwise None.
    """
    if not chunk_ids:
        return []

    if get_hi_rag_config().vdb_type == "lancedb":
        raise NotImplementedError("Lancedb is not supported yet")

    elif get_hi_rag_config().vdb_type == "pgvector":
        try:
            # Create PGVector instance
            vdb = PGVector.create(
                embedding_func=None,
                strategy_provider=RetrievalStrategyProvider(),
                vector_type="halfvec",
            )

            base_chunks = await vdb.query_by_keys(
                key_value=chunk_ids,
                workspace_id=workspace_id or "",
                knowledge_base_id=knowledge_base_id or "",
                table_name="Chunks",
                key_column="documentKey",
            )

            return base_chunks

        except Exception as e:
            log_error_info(
                logging.ERROR, f"Failed to get chunks info for ids={chunk_ids}", e
            )
            return []


async def get_table_info_by_scope(
    table_name: str,
    knowledge_base_id: str,
    workspace_id: str,
) -> list[dict[str, Any]]:
    """Get table info by scope (knowledgeBaseId and workspaceId).

    Args:
        table_name: The name of the table to get info for
        knowledge_base_id: The id of the knowledge base that the table is from
        workspace_id: The id of the workspace that the table is from

    Returns:
        A list of dicts of the table rows if found, otherwise an empty list.
    """
    if not knowledge_base_id or not workspace_id:
        raise ValueError("knowledge_base_id and workspace_id are required")

    results: list[dict[str, Any]] = []

    if get_hi_rag_config().vdb_type == "lancedb":
        raise NotImplementedError("Lancedb is not supported yet")

    elif get_hi_rag_config().vdb_type == "pgvector":
        try:
            vdb = PGVector.create(
                embedding_func=None,
                strategy_provider=RetrievalStrategyProvider(),
                vector_type="halfvec",
            )
            results = await vdb.query_by_keys(
                key_value=[],
                workspace_id=workspace_id,
                knowledge_base_id=knowledge_base_id,
                table_name=table_name,
                key_column="documentKey",
                columns_to_select=None,
                limit=None,
            )
            return results
        except Exception as e:
            log_error_info(
                logging.ERROR,
                f"Failed to get chunk info by scope (pgvector) kb={knowledge_base_id}, ws={workspace_id}",
                e,
            )
            return results


async def get_chunk_info_by_scope(
    knowledge_base_id: str,
    workspace_id: str,
) -> list[dict[str, Any]]:
    """Get chunk info by scope (knowledgeBaseId and workspaceId).

    Args:
        knowledge_base_id: The id of the knowledge base that the chunk is from
        workspace_id: The id of the workspace that the chunk is from

    Returns:
        A list of dicts of the chunk rows if found, otherwise an empty list.
    """
    return await get_table_info_by_scope(
        table_name="Chunks",
        knowledge_base_id=knowledge_base_id,
        workspace_id=workspace_id,
    )


async def get_item_info_by_scope(
    knowledge_base_id: str,
    workspace_id: str,
) -> list[dict[str, Any]]:
    """Get item info by scope (knowledgeBaseId and workspaceId)."""
    return await get_table_info_by_scope(
        table_name="Items",
        knowledge_base_id=knowledge_base_id,
        workspace_id=workspace_id,
    )


async def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="LanceDB Vector Database Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --overview                        # Show database overview
  %(prog)s --list-tables                     # List all tables
  %(prog)s --table chunks --output data.csv  # Export specific table
  %(prog)s --export all --output-dir exports # Export all tables
        """,
    )

    parser.add_argument(
        "--db-path", default="kb/hirag.db", help="LanceDB database path"
    )

    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--overview", action="store_true", help="Show database overview (default)"
    )
    actions.add_argument("--list-tables", action="store_true", help="List all tables")
    actions.add_argument(
        "--export",
        choices=["all", "chunks", "entities", "relations"],
        help="Export data by type",
    )
    actions.add_argument("--table", help="Export specific table")

    parser.add_argument("--output", help="Output file path (for single table export)")
    parser.add_argument(
        "--output-dir",
        default="exports",
        help="Export directory for batch operations (default: exports)",
    )
    parser.add_argument(
        "--exclude-columns",
        nargs="+",
        default=["vector"],
        help="Columns to exclude from export (default: vector)",
    )
    parser.add_argument("--limit", type=int, help="Maximum rows to export")

    args = parser.parse_args()

    if not any([args.list_tables, args.export, args.table]):
        args.overview = True

    try:
        async with VDBManager(args.db_path) as vdb:
            if args.list_tables:
                tables = await vdb.list_tables()
                console.print("[bold]📋 Available Tables:[/bold]")
                for table in sorted(tables):
                    console.print(f"  • {table}")

            elif args.overview:
                await vdb.display_overview()

            elif args.export or args.table:
                await export_handler(vdb, args)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
    except Exception as e:
        log_error_info(logging.ERROR, "Unhandled error", e)
        console.print(f"[red]❌ Error: {e}[/red]")
        raise SystemExit(1)


async def export_handler(vdb: VDBManager, args) -> None:
    """Handle export operations"""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tables = await vdb.list_tables()
    exclude_columns = set(args.exclude_columns) if args.exclude_columns else None

    if args.table:
        if args.table not in tables:
            console.print(f"[red]❌ Table '{args.table}' not found[/red]")
            console.print(f"Available: {', '.join(sorted(tables))}")
            return
        target_tables = [args.table]
    elif args.export == "all":
        target_tables = tables
    else:
        pattern = args.export.lower()
        target_tables = [t for t in tables if pattern in t.lower()]

        if not target_tables:
            console.print(f"[red]❌ No tables matching '{args.export}'[/red]")
            return

    console.print(f"[bold]Exporting {len(target_tables)} table(s)...[/bold]")

    for table_name in target_tables:
        output_file = args.output or f"{args.output_dir}/{table_name}.csv"
        await vdb.export_table_to_csv(
            table_name, output_file, exclude_columns=exclude_columns, limit=args.limit
        )


if __name__ == "__main__":
    asyncio.run(main())
