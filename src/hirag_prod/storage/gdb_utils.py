#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from hirag_prod.storage.networkx import NetworkXGDB

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
console = Console()


class GDBManager:
    """## Graph Database Utilities
    (now use [networkx](https://networkx.org/) as the graph database)

    ### Database Path:
    By default, the tool looks for the graph database at `kb/hirag.gpickle`. You can specify a different
    database path using the `--db-path` parameter:

    `python -m src.hirag_prod.storage.gdb_utils --db-path /path/to/your/graph.gpickle --overview`

    ### Command Line Usage:
    - Basic operations (uses default db: `kb/hirag.gpickle`)
        - `python -m src.hirag_prod.storage.gdb_utils --help`
        - `python -m src.hirag_prod.storage.gdb_utils --overview`
        - `python -m src.hirag_prod.storage.gdb_utils --stats`

    - Custom database path
        - `python -m src.hirag_prod.storage.gdb_utils --db-path /path/to/custom_graph.gpickle --overview`

    - Export nodes and edges
        - `python -m src.hirag_prod.storage.gdb_utils --export nodes --output nodes.csv`
        - `python -m src.hirag_prod.storage.gdb_utils --export edges --output edges.csv`
        - `python -m src.hirag_prod.storage.gdb_utils --export all --output-dir exports`

    - Export with custom database and row limit
        - `python -m src.hirag_prod.storage.gdb_utils --db-path /path/to/graph.gpickle --export nodes --output nodes.csv --limit 100`

    - Query specific node neighbors
        - `python -m src.hirag_prod.storage.gdb_utils --query-node "ent-xxx" --output neighbors.csv`

    ### Output Options:
    - `--output`: Specify output file for single export (used with `--export` or `--query-node`)
    - `--output-dir`: Specify directory for batch export (used with `--export all`)
    - `--limit`: Maximum number of rows to export

    ### Module Usage:
    ```python
    from gdb_utils import GDBManager

    async with GDBManager("path/to/graph.gpickle") as gdb:
        # Get graph stats
        stats = await gdb.get_graph_stats()

        # Export nodes/edges
        await gdb.export_nodes_to_csv("nodes.csv")
        await gdb.export_edges_to_csv("edges.csv")

        # Query node neighbors
        neighbors = await gdb.query_node_neighbors("ent-xxx")
    ```
    """

    def __init__(
        self,
        db_path: str = "kb/hirag.gpickle",
        llm_func=None,
        llm_model_name: str = "gpt-4o-mini",
    ):
        """
        Initialize GDB Manager

        Args:
            db_path: Path to NetworkX graph database gpickle file
            llm_func: LLM function for summarization (optional for read-only operations)
            llm_model_name: LLM model name for summarization
        """
        self.db_path = Path(db_path)
        self.gdb: Optional[NetworkXGDB] = None
        self.llm_func = llm_func
        self.llm_model_name = llm_model_name

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self) -> None:
        """Establish graph database connection"""
        if not self.db_path.exists():
            logger.warning(
                f"Graph database not found: {self.db_path}, creating new graph"
            )

        try:
            # TODO: utilize llm func to analyze the graph?
            # Use a dummy LLM function now
            llm_func = self.llm_func or (lambda x: x)

            self.gdb = NetworkXGDB.create(
                path=str(self.db_path),
                llm_func=llm_func,
                llm_model_name=self.llm_model_name,
            )
            logger.info(f"Connected to Graph DB: {self.db_path}")
        except Exception as e:
            logger.error(f"Graph DB connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Close graph database connection"""
        if self.gdb:
            await self.gdb.clean_up()
            self.gdb = None
            logger.info("Disconnected from Graph DB")

    async def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics

        Returns:
            Dictionary with graph metadata and statistics
        """
        try:
            import networkx as nx

            graph = self.gdb.graph

            # Basic graph metrics
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()

            # Degree statistics
            degrees = dict(graph.degree())
            avg_degree = sum(degrees.values()) / num_nodes if num_nodes > 0 else 0
            max_degree = max(degrees.values()) if degrees else 0
            min_degree = min(degrees.values()) if degrees else 0

            # Component analysis - handle both directed and undirected graphs
            if nx.is_directed(graph):
                # For directed graphs, use weakly connected components
                components = list(nx.weakly_connected_components(graph))
                is_connected = num_nodes > 0 and len(components) == 1
            else:
                # For undirected graphs, use connected components
                components = list(nx.connected_components(graph))
                is_connected = num_nodes > 0 and len(components) == 1

            num_components = len(components)

            # Node types analysis
            node_types = {}
            for node_id, node_data in graph.nodes(data=True):
                if node_id.startswith("chunk-"):
                    entity_type = "CHUNK"
                else:
                    entity_type = node_data.get("entity_type", "UNKNOWN")
                node_types[entity_type] = node_types.get(entity_type, 0) + 1

            # Calculate density based on graph type
            if nx.is_directed(graph):
                # For directed graphs: edges / (n * (n-1))
                max_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
            else:
                # For undirected graphs: edges / (n * (n-1) / 2)
                max_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 1

            density = round(num_edges / max_edges, 4) if max_edges > 0 else 0

            return {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "avg_degree": round(avg_degree, 2),
                "max_degree": max_degree,
                "min_degree": min_degree,
                "is_connected": is_connected,
                "num_components": num_components,
                "node_types": node_types,
                "density": density,
                "graph_type": "directed" if nx.is_directed(graph) else "undirected",
            }
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {"error": str(e)}

    async def get_nodes_sample(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample nodes for display"""
        try:
            nodes = list(self.gdb.graph.nodes(data=True))[:limit]
            return [{"id": node_id, **node_data} for node_id, node_data in nodes]
        except Exception as e:
            logger.error(f"Failed to get nodes sample: {e}")
            return []

    async def get_edges_sample(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample edges for display"""
        try:
            edges = list(self.gdb.graph.edges(data=True))[:limit]
            return [
                {"source": source, "target": target, **edge_data}
                for source, target, edge_data in edges
            ]
        except Exception as e:
            logger.error(f"Failed to get edges sample: {e}")
            return []

    async def export_nodes_to_csv(
        self, output_file: str, limit: Optional[int] = None
    ) -> bool:
        """
        Export nodes to CSV file

        Args:
            output_file: Output CSV path
            limit: Maximum nodes to export

        Returns:
            Success status
        """
        try:
            nodes_data = []
            graph_nodes = list(self.gdb.graph.nodes(data=True))

            if limit:
                graph_nodes = graph_nodes[:limit]

            for node_id, node_data in graph_nodes:
                # Flatten node data for CSV
                row = {"id": node_id}
                for key, value in node_data.items():
                    row[key] = self._serialize_value(value)
                nodes_data.append(row)

            # Write to CSV
            if nodes_data:
                import pandas as pd

                df = pd.DataFrame(nodes_data)
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False, encoding="utf-8")

                console.print(
                    f"[green]✅ Exported {len(nodes_data)} nodes to {output_file}[/green]"
                )
                return True
            else:
                console.print("[yellow]⚠️  No nodes to export[/yellow]")
                return False

        except Exception as e:
            logger.error(f"Nodes export failed: {e}")
            console.print(f"[red]❌ Nodes export failed: {e}[/red]")
            return False

    async def export_edges_to_csv(
        self, output_file: str, limit: Optional[int] = None
    ) -> bool:
        """
        Export edges to CSV file

        Args:
            output_file: Output CSV path
            limit: Maximum edges to export

        Returns:
            Success status
        """
        try:
            edges_data = []
            graph_edges = list(self.gdb.graph.edges(data=True))

            if limit:
                graph_edges = graph_edges[:limit]

            for source, target, edge_data in graph_edges:
                # Flatten edge data for CSV
                row = {"source": source, "target": target}
                for key, value in edge_data.items():
                    row[key] = self._serialize_value(value)
                edges_data.append(row)

            # Write to CSV
            if edges_data:
                import pandas as pd

                df = pd.DataFrame(edges_data)
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False, encoding="utf-8")

                console.print(
                    f"[green]✅ Exported {len(edges_data)} edges to {output_file}[/green]"
                )
                return True
            else:
                console.print("[yellow]⚠️  No edges to export[/yellow]")
                return False

        except Exception as e:
            logger.error(f"Edges export failed: {e}")
            console.print(f"[red]❌ Edges export failed: {e}[/red]")
            return False

    async def query_node_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Query a node and its neighbors"""
        try:
            if node_id not in self.gdb.graph.nodes:
                console.print(f"[red]❌ Node '{node_id}' not found[/red]")
                return []

            neighbors, relations = await self.gdb.query_one_hop(node_id)

            # Convert to serializable format
            result = []
            for neighbor in neighbors:
                result.append(
                    {
                        "id": neighbor.id,
                        "page_content": neighbor.page_content,
                        **{
                            k: self._serialize_value(v)
                            for k, v in neighbor.metadata.__dict__.items()
                        },
                    }
                )

            return result
        except Exception as e:
            logger.error(f"Failed to query node neighbors: {e}")
            return []

    async def export_node_neighbors_to_csv(
        self, node_id: str, output_file: str
    ) -> bool:
        """Export node neighbors to CSV"""
        try:
            neighbors = await self.query_node_neighbors(node_id)

            if neighbors:
                import pandas as pd

                df = pd.DataFrame(neighbors)
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False, encoding="utf-8")

                console.print(
                    f"[green]✅ Exported {len(neighbors)} neighbors of '{node_id}' to {output_file}[/green]"
                )
                return True
            else:
                console.print(
                    f"[yellow]⚠️  No neighbors found for node '{node_id}'[/yellow]"
                )
                return False

        except Exception as e:
            logger.error(f"Neighbor export failed: {e}")
            console.print(f"[red]❌ Neighbor export failed: {e}[/red]")
            return False

    @staticmethod
    def _serialize_value(value: Any) -> str:
        """Serialize complex values for CSV export"""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _format_value(self, value: Any, max_length: int = 50) -> str:
        """Format value for display"""
        if isinstance(value, list) and len(value) > 3:
            return f"<list[{len(value)}]>"

        str_val = str(value)
        if len(str_val) > max_length:
            return str_val[:max_length] + "..."
        return str_val

    async def display_simple_overview(self) -> None:
        """Display simplified graph database overview"""
        stats = await self.get_graph_stats()

        if "error" in stats:
            console.print(f"[red]❌ Failed to get graph stats: {stats['error']}[/red]")
            return

        # Main stats panel
        console.print(
            Panel.fit(
                f"[bold cyan]NetworkX Graph Database[/bold cyan]\n"
                f"📍 Path: {self.db_path}\n"
                f"🔗 Nodes: {stats['num_nodes']:,} | Edges: {stats['num_edges']:,}\n"
                f"📊 Density: {stats['density']:.4f} | Components: {stats['num_components']}\n"
                f"📈 Type: {stats['graph_type'].title()} Graph",
                border_style="blue",
            )
        )

        # Node types distribution
        if stats["node_types"]:
            console.print("\n[bold]🏷️  Node Types Distribution:[/bold]")
            for entity_type, count in sorted(
                stats["node_types"].items(), key=lambda x: x[1], reverse=True
            ):
                console.print(f"  • {entity_type}: {count:,}")

    async def display_overview(self) -> None:
        """Display comprehensive graph database overview with detailed statistics"""
        stats = await self.get_graph_stats()

        if "error" in stats:
            console.print(f"[red]❌ Failed to get graph stats: {stats['error']}[/red]")
            return

        # Main stats panel
        console.print(
            Panel.fit(
                f"[bold cyan]NetworkX Graph Database[/bold cyan]\n"
                f"📍 Path: {self.db_path}\n"
                f"🔗 Nodes: {stats['num_nodes']:,} | Edges: {stats['num_edges']:,}\n"
                f"📊 Density: {stats['density']:.4f} | Components: {stats['num_components']}\n"
                f"📈 Type: {stats['graph_type'].title()} Graph",
                border_style="blue",
            )
        )

        # Detailed stats table
        table = Table(title="📈 Graph Statistics", show_header=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green")

        table.add_row("Total Nodes", f"{stats['num_nodes']:,}")
        table.add_row("Total Edges", f"{stats['num_edges']:,}")
        table.add_row("Graph Type", stats["graph_type"].title())
        table.add_row("Average Degree", str(stats["avg_degree"]))
        table.add_row("Max Degree", str(stats["max_degree"]))
        table.add_row("Min Degree", str(stats["min_degree"]))
        table.add_row("Density", f"{stats['density']:.4f}")
        table.add_row("Is Connected", "✅ Yes" if stats["is_connected"] else "❌ No")
        table.add_row("Components", str(stats["num_components"]))

        console.print(table)

        # Node types distribution
        if stats["node_types"]:
            console.print("\n[bold]🏷️  Node Types Distribution:[/bold]")
            for entity_type, count in sorted(
                stats["node_types"].items(), key=lambda x: x[1], reverse=True
            ):
                console.print(f"  • {entity_type}: {count:,}")

        # Sample data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading samples...", total=2)

            # Sample nodes
            progress.update(task, description="Loading node samples...")
            nodes_sample = await self.get_nodes_sample(3)
            progress.advance(task)

            if nodes_sample:
                console.print("\n[bold]🔍 Sample Nodes:[/bold]")
                for i, node in enumerate(nodes_sample, 1):
                    console.print(f"  {i}. [cyan]{node['id']}[/cyan]")
                    for key, value in list(node.items())[
                        1:4
                    ]:  # Show first 3 attributes
                        formatted = self._format_value(value)
                        console.print(f"     • {key}: {formatted}")
                    if len(node) > 4:
                        console.print(f"     ... and {len(node) - 4} more attributes")

            # Sample edges
            progress.update(task, description="Loading edge samples...")
            edges_sample = await self.get_edges_sample(3)
            progress.advance(task)

            if edges_sample:
                console.print("\n[bold]🔗 Sample Edges:[/bold]")
                for i, edge in enumerate(edges_sample, 1):
                    console.print(
                        f"  {i}. [cyan]{edge['source']}[/cyan] → [cyan]{edge['target']}[/cyan]"
                    )
                    for key, value in list(edge.items())[
                        2:4
                    ]:  # Show first 2 edge attributes
                        formatted = self._format_value(value)
                        console.print(f"     • {key}: {formatted}")


async def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="NetworkX Graph Database Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --overview                          # Show simple graph overview
  %(prog)s --stats                             # Show detailed statistics of graph
  %(prog)s --export nodes --output nodes.csv   # Export nodes
  %(prog)s --export edges --output edges.csv   # Export edges
  %(prog)s --export all --output-dir exports   # Export both to directory
  %(prog)s --query-node "ent-xxx"              # Query node neighbors by node id
        """,
    )

    parser.add_argument(
        "--db-path", default="kb/hirag.gpickle", help="Graph database path"
    )

    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--overview", action="store_true", help="Show simple graph overview (default)"
    )
    actions.add_argument(
        "--stats", action="store_true", help="Show detailed statistics of graph"
    )
    actions.add_argument(
        "--export", choices=["all", "nodes", "edges"], help="Export data"
    )
    actions.add_argument("--query-node", help="Query specific node and its neighbors")

    parser.add_argument("--output", help="Output file path")
    parser.add_argument(
        "--output-dir", default="exports", help="Export directory (default: exports)"
    )
    parser.add_argument("--limit", type=int, help="Maximum rows to export")

    args = parser.parse_args()

    # Default to overview if no action specified
    if not any([args.stats, args.export, args.query_node]):
        args.overview = True

    try:
        async with GDBManager(args.db_path) as gdb:
            if args.overview:
                await gdb.display_simple_overview()
            elif args.stats:
                await gdb.display_overview()
            elif args.export:
                await export_handler(gdb, args)
            elif args.query_node:
                output_file = args.output or f"{args.query_node}_neighbors.csv"
                await gdb.export_node_neighbors_to_csv(args.query_node, output_file)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        logger.exception("Unhandled error")
        raise SystemExit(1)


async def export_handler(gdb: GDBManager, args) -> None:
    """Handle export operations"""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.export == "nodes":
        output_file = args.output or f"{args.output_dir}/nodes.csv"
        await gdb.export_nodes_to_csv(output_file, limit=args.limit)
    elif args.export == "edges":
        output_file = args.output or f"{args.output_dir}/edges.csv"
        await gdb.export_edges_to_csv(output_file, limit=args.limit)
    elif args.export == "all":
        console.print("[bold]Exporting nodes and edges...[/bold]")
        nodes_file = f"{args.output_dir}/nodes.csv"
        edges_file = f"{args.output_dir}/edges.csv"

        await asyncio.gather(
            gdb.export_nodes_to_csv(nodes_file, limit=args.limit),
            gdb.export_edges_to_csv(edges_file, limit=args.limit),
        )


if __name__ == "__main__":
    asyncio.run(main())
