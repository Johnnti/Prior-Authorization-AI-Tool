#!/usr/bin/env python3
"""
Prior Authorization AI Tool - Main Entry Point

This script processes PA forms by extracting information from referral packages
and filling out the corresponding PA forms using AI-powered extraction.

Usage:
    # Process all patient folders
    python main.py --all
    
    # Process specific folder
    python main.py --folder Adbulla
    
    # Process with vision AI (for scanned documents)
    python main.py --folder Adbulla --vision
    
    # Run as API server
    python main.py --server
    
    # Set API key via command line
    python main.py --all --openai-key YOUR_KEY
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.processing_service import PAProcessingService
from src.models import FieldStatus


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def print_banner():
    """Print application banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          Prior Authorization AI Tool v1.0.0                  ║
║     AI-Powered PA Form Processing & Auto-Fill                ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_result_summary(result):
    """Print a formatted summary of processing results."""
    print("\n" + "=" * 60)
    print(f"📁 Patient Folder: {result.patient_folder}")
    print("=" * 60)
    
    if result.success:
        print(f"✅ Status: SUCCESS")
        print(f"⏱️  Processing Time: {result.processing_time:.2f}s")
        print(f"📄 Output: {result.output_path}")
        
        # Filled fields
        if result.filled_fields:
            print(f"\n✅ FILLED FIELDS ({len(result.filled_fields)}):")
            print("-" * 40)
            for field in result.filled_fields:
                confidence_bar = "█" * int(field.confidence * 10) + "░" * (10 - int(field.confidence * 10))
                print(f"  • {field.name}: {field.value}")
                print(f"    Confidence: [{confidence_bar}] {field.confidence:.0%}")
        
        # Uncertain fields
        if result.uncertain_fields:
            print(f"\n⚠️  UNCERTAIN FIELDS ({len(result.uncertain_fields)}):")
            print("-" * 40)
            for field in result.uncertain_fields:
                print(f"  • {field.name}: {field.value} (confidence: {field.confidence:.0%})")
        
        # Unfilled fields
        if result.unfilled_fields:
            print(f"\n❌ FIELDS NOT FOUND ({len(result.unfilled_fields)}):")
            print("-" * 40)
            for field in result.unfilled_fields:
                print(f"  • {field.name}")
        
        # Summary statistics
        total = len(result.filled_fields) + len(result.uncertain_fields) + len(result.unfilled_fields)
        filled_pct = len(result.filled_fields) / total * 100 if total > 0 else 0
        print(f"\n📊 COMPLETION: {filled_pct:.1f}% ({len(result.filled_fields)}/{total} fields)")
        
    else:
        print(f"❌ Status: FAILED")
        print(f"Error: {result.error_message}")
    
    print("=" * 60 + "\n")


def print_batch_summary(batch_result):
    """Print a formatted summary of batch processing results."""
    summary = batch_result.get_summary()
    
    print("\n" + "═" * 60)
    print("📊 BATCH PROCESSING SUMMARY")
    print("═" * 60)
    print(f"Total Processed: {summary['total_processed']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total Time: {summary['total_time']:.2f}s")
    print("═" * 60 + "\n")


def list_available_folders(service):
    """List available patient folders."""
    folders = service.get_available_folders()
    
    print("\n📁 Available Patient Folders:")
    print("-" * 40)
    
    for folder in folders:
        status = "✅ Ready" if folder["ready"] else "⚠️  Missing files"
        pa_status = "✓" if folder["has_pa_form"] else "✗"
        ref_status = "✓" if folder["has_referral_package"] else "✗"
        
        print(f"  {folder['name']}: {status}")
        print(f"    PA Form: {pa_status}  |  Referral Package: {ref_status}")
    
    print("-" * 40)
    print(f"Total: {len(folders)} folders ({sum(1 for f in folders if f['ready'])} ready)\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prior Authorization AI Tool - Process PA forms using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --list                    List available folders
  python main.py --folder Adbulla          Process single folder
  python main.py --all                     Process all folders
  python main.py --server                  Start API server
  python main.py --all --openai-key KEY    Process with API key
        """
    )
    
    # Processing options
    parser.add_argument("--folder", "-f", type=str, help="Process a specific patient folder")
    parser.add_argument("--all", "-a", action="store_true", help="Process all patient folders")
    parser.add_argument("--list", "-l", action="store_true", help="List available folders")
    parser.add_argument("--vision", "-v", action="store_true", default=True,
                       help="Use vision AI for scanned documents (default: True)")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision AI")
    parser.add_argument("--parallel", "-p", action="store_true", help="Process folders in parallel")
    
    # API server options
    parser.add_argument("--server", "-s", action="store_true", help="Run as API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    
    # Configuration options
    parser.add_argument("--openai-key", type=str, help="OpenAI API key")
    parser.add_argument("--anthropic-key", type=str, help="Anthropic API key")
    parser.add_argument("--provider", type=str, choices=["openai", "anthropic"],
                       default="openai", help="AI provider to use")
    parser.add_argument("--input-dir", type=str, help="Input directory path")
    parser.add_argument("--output-dir", type=str, help="Output directory path")
    
    # Logging options
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Print banner
    print_banner()
    
    # Update configuration from arguments
    if args.openai_key:
        config.ai.openai_api_key = args.openai_key
    if args.anthropic_key:
        config.ai.anthropic_api_key = args.anthropic_key
    if args.provider:
        config.ai.provider = args.provider
    if args.input_dir:
        config.input_dir = Path(args.input_dir)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    
    use_vision = args.vision and not args.no_vision
    
    # Run API server
    if args.server:
        print(f"🚀 Starting API server on {args.host}:{args.port}...")
        from src.api import run_server
        run_server(host=args.host, port=args.port)
        return
    
    # Initialize service
    service = PAProcessingService(config)
    
    # List folders
    if args.list:
        list_available_folders(service)
        return
    
    # Process single folder
    if args.folder:
        folder_path = config.input_dir / args.folder
        if not folder_path.exists():
            print(f"❌ Error: Folder '{args.folder}' not found in {config.input_dir}")
            sys.exit(1)
        
        print(f"🔄 Processing folder: {args.folder}")
        result = service.process_patient_folder(folder_path, use_vision=use_vision)
        print_result_summary(result)
        
        if not result.success:
            sys.exit(1)
        return
    
    # Process all folders
    if args.all:
        print(f"🔄 Processing all folders in: {config.input_dir}")
        batch_result = service.process_all_folders(parallel=args.parallel)
        
        # Print individual results
        for result in batch_result.results:
            print_result_summary(result)
        
        # Print batch summary
        print_batch_summary(batch_result)
        
        # Exit with error if any failed
        if any(not r.success for r in batch_result.results):
            sys.exit(1)
        return
    
    # No action specified - show help
    parser.print_help()
    print("\n💡 Tip: Use --list to see available folders, or --all to process everything.")


if __name__ == "__main__":
    main()
