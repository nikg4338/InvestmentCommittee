#!/usr/bin/env python3
"""
Enhanced Automated Batch Training Script for Investment Committe    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🔄 Collecting LEAK-FREE data for batch {batch_num} (attempt {attempt}/{max_retries})...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0 and Path(data_file).exists():
                logger.info(f"✅ LEAK-FREE data collection successful for batch {batch_num}")
                return data_file
            else:
                logger.warning(f"Leak-free data collection attempt {attempt} failed (code {result.returncode}). STDERR: {result.stderr[:300]}")
        except subprocess.TimeoutExpired:
            logger.warning(f"Leak-free data collection timeout (attempt {attempt})")
        except Exception as e:
            logger.warning(f"Leak-free data collection error (attempt {attempt}): {e}")

    logger.error(f"Failed to collect leak-free data for batch {batch_num} after {max_retries} attempts")
    return None===================================================

This orchestrates batch data collection, training, telemetry extraction, and report organization.

Usage examples:
  python train_all_batches.py                    # Process all non-empty batches from filtered_iex_batches.json
  python train_all_batches.py --batch 1          # Process a single batch
  python train_all_batches.py --start 1 --end 5  # Process a range
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure logs directory exists before configuring logging
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/batch_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def ensure_directories() -> None:
    """Ensure required directories exist."""
    dirs = ['logs', 'reports', 'data', 'models/saved']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info(f"Created directories: {', '.join(dirs)}")


def load_batch_data(batch_file: str = 'filtered_iex_batches.json') -> Dict[str, Any]:
    """Load batch information from filtered_iex_batches.json."""
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'batches' not in data:
            raise KeyError("Missing 'batches' key in batch file")
        return data
    except FileNotFoundError:
        logger.error(f"Batch file not found: {batch_file}")
        return {"batches": {}}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {batch_file}: {e}")
        return {"batches": {}}


def get_non_empty_batches(batch_data: Dict[str, Any]) -> List[int]:
    """Get list of batch numbers that contain at least one symbol."""
    non_empty: List[int] = []
    for batch_name, symbols in batch_data.get('batches', {}).items():
        try:
            batch_num = int(batch_name.replace('batch_', '').strip())
        except Exception:
            continue
        if isinstance(symbols, list) and len(symbols) > 0:
            non_empty.append(batch_num)
    non_empty.sort()
    logger.info(f"Found {len(non_empty)} non-empty batches: {non_empty}")
    return non_empty


def collect_batch_data(batch_num: int, max_retries: int = 3, timeout: int = 1200) -> Optional[str]:
    """Collect LEAK-FREE data for a specific batch using the fixed data collection. Returns CSV path or None."""
    data_file = f"data/leak_free_batch_{batch_num}_data.csv"
    
    # CRITICAL FIX: ALWAYS collect fresh data - NO file reuse to prevent leakage
    logger.info(f"� COLLECTING FRESH LEAK-FREE DATA (no file reuse to prevent leakage)")
    
    # Remove existing file to force fresh collection
    if Path(data_file).exists():
        logger.info(f"�️  Removing existing file to ensure fresh collection: {data_file}")
        Path(data_file).unlink()

    cmd = [sys.executable, 'data_collection_alpaca_fixed.py', '--batch', str(batch_num), '--output', Path(data_file).name]
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🔄 Collecting validated data for batch {batch_num} (attempt {attempt}/{max_retries})…")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0 and Path(data_file).exists():
                logger.info(f"✅ Validated data collected for batch {batch_num} → {data_file}")
                return data_file
            else:
                logger.warning(f"Data collection attempt {attempt} failed (code {result.returncode}). STDERR: {result.stderr[:300]}")
        except subprocess.TimeoutExpired:
            logger.warning(f"Data collection timeout (attempt {attempt})")
        except Exception as e:
            logger.warning(f"Data collection error (attempt {attempt}): {e}")

    logger.error(f"Failed to collect validated data for batch {batch_num} after {max_retries} attempts")
    return None


def train_batch_models(
    batch_num: int,
    data_file: str,
    config: str = "extreme_imbalance",
    optuna_trials: int = 15,
    timeout: int = 2400,
    disable_enhancements: bool = False,
) -> bool:
    """Train models for a specific batch using train_models.py and extract telemetry."""
    
    # CRITICAL: Ensure we're using leak-free data
    if not data_file.startswith("data/leak_free_"):
        logger.error(f"🚨 REFUSING TO TRAIN ON POTENTIALLY LEAKY DATA: {data_file}")
        logger.error("Only files starting with 'data/leak_free_' are allowed!")
        return False
    
    if disable_enhancements:
        logger.info(f"Training models for batch {batch_num} (Standard Pipeline)…")
    else:
        logger.info(f"Training models for batch {batch_num} (LEAK-FREE Enhanced Pipeline)…")

    run_start = datetime.now()
    trainer_script = 'train_models.py'
    
    # Ensure plots directory exists for plot saving
    os.makedirs('plots', exist_ok=True)
    
    cmd_args = [
        sys.executable, '-u', trainer_script,
        '--data-file', data_file,
        '--config', config,
        '--target-column', 'target',
        '--save-plots',  # Force plot saving
        '--export-results',
        '--log-level', 'INFO',
        '--batch-id', str(batch_num),
        '--optuna-trials', str(optuna_trials),
        '--telemetry-json', str(Path('logs') / f'telemetry_batch_{batch_num}.json'),
    ]
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    try:
        # Don't capture output - let training script output directly
        # This allows logging to work properly in subprocesses
        result = subprocess.run(cmd_args, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        logger.error(f"Training timeout for batch {batch_num} ({timeout//60} minutes)")
        return False
    except Exception as e:
        logger.error(f"Training error for batch {batch_num}: {e}")
        return False

    if result.returncode != 0:
        enhancement_type = "Enhanced" if not disable_enhancements else "Standard"
        logger.error(f"{enhancement_type} training failed for batch {batch_num} (exit code: {result.returncode})")
        return False

    enhancement_type = "Standard" if disable_enhancements else "Enhanced"
    logger.info(f"{enhancement_type} training completed successfully for batch {batch_num}")

    # Since we don't capture stdout, check for result files and telemetry directly
    # Wait a moment for files to be written
    import time
    time.sleep(1)
    
    # Check if training produced any files after run_start
    telemetry_json_path = Path('logs') / f'telemetry_batch_{batch_num}.json'
    recent_results = []
    recent_telemetry = False
    
    # Check for recent detailed_results files
    for p in Path('logs').glob('detailed_results_*.json'):
        try:
            if datetime.fromtimestamp(p.stat().st_mtime) >= run_start:
                recent_results.append(p)
        except Exception:
            pass
    
    # Check if telemetry JSON was updated recently
    if telemetry_json_path.exists():
        try:
            if datetime.fromtimestamp(telemetry_json_path.stat().st_mtime) >= run_start:
                recent_telemetry = True
        except Exception:
            pass
    
    if recent_results or recent_telemetry:
        logger.info(f"Training produced {len(recent_results)} result files and telemetry_recent={recent_telemetry}")
    else:
        logger.warning("Training subprocess completed but produced no recent files - may have exited silently")
        logger.warning("To run training successfully, try: python train_models.py --data-file data/batch_1_data.csv --config default --target-column target --export-results")

    # Extract telemetry markers for reports
    telemetry_markers = [
        "Dynamic ensemble weights applied",
        "PR-AUC gate:",
        "Applied PR-AUC–prioritized dynamic weights",
        "Batch signal quality PASSED",
        "Batch signal quality check FAILED",
        "TELEMETRY|",
    ]
    telemetry_lines: List[str] = []
    
    # Since stdout wasn't captured, search training.log for recent telemetry
    logger.info("Searching training.log for recent telemetry markers...")
    try:
        training_log_path = Path('logs') / 'training.log'
        if training_log_path.exists():
            with open(training_log_path, 'r', encoding='utf-8', errors='ignore') as lf:
                for line in lf:
                    try:
                        ts_str = line.split(' - ', 1)[0].strip()
                        ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')
                        if ts < run_start:
                            continue
                    except Exception:
                        pass
                    if any(marker in line for marker in telemetry_markers):
                        telemetry_lines.append(line.rstrip())
            logger.info(f"Found {len(telemetry_lines)} telemetry lines in training.log")
    except Exception as e:
        logger.debug(f"Telemetry search failed: {e}")

    # Check for telemetry JSON file (primary source for metrics)
    telemetry_json_path = Path('logs') / f'telemetry_batch_{batch_num}.json'
    telemetry_summary = "No telemetry JSON available"
    
    if telemetry_json_path.exists():
        try:
            with open(telemetry_json_path, 'r', encoding='utf-8') as f:
                telemetry_data = json.load(f)
            
            # Extract key metrics for logging
            pr_auc = telemetry_data.get('pr_auc_meta', 0.0)
            gate = telemetry_data.get('gate', 'UNKNOWN')
            weights = telemetry_data.get('dynamic_weights', {})
            
            telemetry_summary = f"PR-AUC: {pr_auc:.3f}, Gate: {gate}, Models: {len(weights)}"
            logger.info(f"✓ Telemetry JSON found: {telemetry_summary}")
            
        except Exception as e:
            logger.warning(f"Failed to parse telemetry JSON: {e}")

    if telemetry_lines:
        os.makedirs('logs', exist_ok=True)
        telemetry_path = Path('logs') / f'telemetry_batch_{batch_num}.log'
        with open(telemetry_path, 'w', encoding='utf-8') as tf:
            tf.write(f"Batch {batch_num} Telemetry\n")
            tf.write(f"Timestamp: {datetime.now().isoformat()}\n")
            tf.write(f"Summary: {telemetry_summary}\n\n")
            tf.write("\n".join(telemetry_lines))
        logger.info(f"Wrote telemetry log: {len(telemetry_lines)} lines + summary")
    else:
        logger.info("No telemetry markers found; relying on JSON file and result artifacts")

    # Ensure a structured telemetry JSON exists; synthesize from results if missing
    try:
        telem_json = Path('logs') / f'telemetry_batch_{batch_num}.json'
        if not telem_json.exists():
            candidates = []
            for base in [Path('results'), Path('logs')]:
                if base.exists():
                    for p in base.glob('detailed_results_*.json'):
                        try:
                            if p.stat().st_mtime >= run_start.timestamp():
                                candidates.append(p)
                        except Exception:
                            candidates.append(p)
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                with open(latest, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                ensemble_perf = (data.get('ensemble_performance') or {}).get('meta_model') or {}
                pr_auc_meta = float(ensemble_perf.get('pr_auc', 0.0) or 0.0)
                base_perf = data.get('base_model_performance') or {}
                base_perf_slim = {
                    k: {
                        'pr_auc': float((v or {}).get('pr_auc', 0.0) or 0.0),
                        'roc_auc': float((v or {}).get('roc_auc', 0.0) or 0.0),
                    } for k, v in base_perf.items()
                }
                dyn_weights = data.get('dynamic_weights') or {}
                payload: Dict[str, Any] = {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'batch_id': str(batch_num),
                    'gate_threshold': 0.05,
                    'gate': 'PASS' if pr_auc_meta >= 0.05 else 'FAIL',
                    'pr_auc_meta': pr_auc_meta,
                    'meta_model_pr_auc_reported': pr_auc_meta,
                    'dynamic_weights': dyn_weights,
                    'base_model_performance': base_perf_slim,
                    'source': str(latest),
                }
            else:
                payload = {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'batch_id': str(batch_num),
                    'note': 'No current-run detailed_results found; minimal telemetry emitted',
                    'source': 'orchestrator',
                }
            with open(telem_json, 'w', encoding='utf-8') as jf:
                json.dump(payload, jf, indent=2)
            logger.info(f"Wrote telemetry JSON → {telem_json}")
    except Exception as e:
        logger.debug(f"Failed to synthesize telemetry JSON: {e}")

    return True


def organize_batch_results(batch_num: int, data_file: Optional[str], run_start: Optional[datetime] = None) -> bool:
    """Organize results, plots, logs, and data into reports/batch_<n>."""
    try:
        batch_reports_dir = Path('reports') / f'batch_{batch_num}'
        batch_reports_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Organizing results for batch {batch_num} in {batch_reports_dir}")

        # Move results directory if produced
        results_dir = Path('results')
        dest_results = batch_reports_dir / 'results'
        if results_dir.exists():
            if dest_results.exists():
                shutil.rmtree(dest_results)
            shutil.move(str(results_dir), str(dest_results))
            logger.info(f"Moved results to {dest_results}")
        else:
            # Collect recent result artifacts from logs
            dest_results.mkdir(parents=True, exist_ok=True)
            copied = 0
            try:
                for p in Path('logs').glob('detailed_results_*.json'):
                    if run_start is None or datetime.fromtimestamp(p.stat().st_mtime) >= run_start:
                        shutil.copy2(p, dest_results / p.name)
                        copied += 1
                # Copy training summaries
                for p in Path('logs').glob('training_summary*.csv'):
                    if run_start is None or datetime.fromtimestamp(p.stat().st_mtime) >= run_start:
                        shutil.copy2(p, dest_results / p.name)
                        copied += 1
                if copied:
                    logger.info(f"Collected {copied} recent result artifacts into {dest_results}")
                else:
                    # Check if there's a valid telemetry JSON even without detailed results
                    telemetry_json = Path('logs') / f'telemetry_batch_{batch_num}.json'
                    has_valid_telemetry = False
                    if telemetry_json.exists():
                        try:
                            with open(telemetry_json, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            pr_auc = data.get('pr_auc_meta', 0.0)
                            gate = data.get('gate', 'UNKNOWN')
                            if pr_auc > 0 and gate in ['PASS', 'FAIL']:
                                has_valid_telemetry = True
                                logger.info(f"Found valid telemetry: PR-AUC={pr_auc:.3f}, Gate={gate}")
                        except Exception:
                            pass
                    
                    if has_valid_telemetry:
                        logger.info("No detailed results but valid telemetry found - creating minimal result summary")
                        (dest_results / 'MINIMAL_RESULTS.txt').write_text(
                            f'Training completed with telemetry but no detailed results artifacts.\n'
                            f'Check logs/training.log and telemetry files for details.\n'
                            f'Telemetry: PR-AUC={pr_auc:.3f}, Gate={gate}\n',
                            encoding='utf-8'
                        )
                    else:
                        logger.warning("No current-run results found; creating EMPTY_RESULTS sentinel")
                        (dest_results / 'EMPTY_RESULTS.txt').write_text(
                            'No results were produced for this run. Check logs/training.log and telemetry logs.',
                            encoding='utf-8'
                        )
            except Exception as e:
                logger.warning(f"Could not collect result artifacts: {e}")

        # Move plots
        plots_moved = False
        if Path('plots').exists():
            dest = batch_reports_dir / 'plots'
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move('plots', dest)
            plots_moved = True
            logger.info(f"Moved plots directory to {dest}")
            
            # List what plots were moved
            try:
                plot_files = list(dest.glob('*'))
                logger.info(f"Plots found: {[p.name for p in plot_files[:10]]}")  # Show first 10
            except Exception as e:
                logger.warning(f"Could not list plot files: {e}")
        else:
            logger.warning("No plots directory found - plots may not have been generated")

        # Copy any performance plots in root directory to this batch folder
        root_plots_found = 0
        patterns = ["performance_comparison_*.png", "confusion_matrices_*.png", "class_distribution_*.png"]
        for pat in patterns:
            for p in glob(pat):
                shutil.move(p, batch_reports_dir / Path(p).name)
                root_plots_found += 1
                logger.info(f"Moved root plot: {Path(p).name}")
        
        if not plots_moved and root_plots_found == 0:
            logger.warning("No plots were found or moved - check if --save-plots is working")

        # Copy training log
        training_log = Path('logs') / 'training.log'
        if training_log.exists():
            shutil.copy2(training_log, batch_reports_dir / f'batch_{batch_num}_training.log')

        # Copy telemetry summary
        telemetry_log = Path('logs') / f'telemetry_batch_{batch_num}.log'
        if telemetry_log.exists():
            shutil.copy2(telemetry_log, batch_reports_dir / f'batch_{batch_num}_telemetry.log')
        # Copy structured telemetry JSON if exists
        telemetry_json = Path('logs') / f'telemetry_batch_{batch_num}.json'
        if telemetry_json.exists():
            shutil.copy2(telemetry_json, batch_reports_dir / f'batch_{batch_num}_telemetry.json')

        # Copy data file
        if data_file and Path(data_file).exists():
            shutil.copy2(data_file, batch_reports_dir / Path(data_file).name)

        create_batch_summary(batch_num, batch_reports_dir)
        logger.info(f"Results organized successfully for batch {batch_num}")
        # Consider a batch failed if results folder has only the empty sentinel
        try:
            entries = list((batch_reports_dir / 'results').glob('*'))
            if len(entries) == 1 and entries[0].name == 'EMPTY_RESULTS.txt':
                logger.error("Batch marked as FAILED due to empty results.")
                return False
            elif len(entries) == 1 and entries[0].name == 'MINIMAL_RESULTS.txt':
                logger.warning("Batch completed with telemetry only (no detailed results) - marked as SUCCESS")
                return True
        except Exception:
            pass
        return True
    except Exception as e:
        logger.error(f"Failed to organize results for batch {batch_num}: {e}")
        return False


def create_batch_summary(batch_num: int, batch_dir: Path) -> None:
    """Create a minimal text summary for the batch in the reports folder."""
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines = [
            f"Batch {batch_num} Summary",
            f"Generated: {now}",
            "",
            "Artifacts:",
            f"- Results: {batch_dir / 'results'}",
            f"- Plots: {batch_dir / 'plots'}",
            f"- Training Log: {batch_dir / f'batch_{batch_num}_training.log'}",
            f"- Telemetry: {batch_dir / f'batch_{batch_num}_telemetry.log'}",
        ]
        with open(batch_dir / 'SUMMARY.txt', 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
    except Exception as e:
        logger.warning(f"Could not write batch summary: {e}")


def process_batch(
    batch_num: int,
    config: str = 'extreme_imbalance',
    optuna_trials: int = 15,
    timeout: int = 2400,
    disable_enhancements: bool = False,
    skip_collection: bool = False,
) -> bool:
    """End-to-end process for a single batch."""
    ensure_directories()

    data_file = f"data/batch_{batch_num}_data.csv"
    if not skip_collection:
        data_file = collect_batch_data(batch_num) or data_file

    if not Path(data_file).exists():
        logger.error(f"Data file not found for batch {batch_num}: {data_file}")
        return False

    run_start = datetime.now()
    ok = train_batch_models(
        batch_num=batch_num,
        data_file=data_file,
        config=config,
        optuna_trials=optuna_trials,
        timeout=timeout,
        disable_enhancements=disable_enhancements,
    )
    org_ok = organize_batch_results(batch_num, data_file if ok else None, run_start=run_start)
    return bool(ok and org_ok)


def create_master_summary(successful_batches: List[int], failed_batches: List[int], total_time: float) -> None:
    """Write a simple master summary under reports/."""
    try:
        reports_dir = Path('reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / 'MASTER_SUMMARY.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"Completed in {total_time:.1f}s\n")
            f.write(f"Successful batches: {successful_batches}\n")
            f.write(f"Failed batches: {failed_batches}\n")
        logger.info(f"Wrote master summary: {path}")
    except Exception as e:
        logger.warning(f"Could not write master summary: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Process and train batches for Investment Committee')
    parser.add_argument('--batch', type=int, help='Single batch number to process')
    parser.add_argument('--start', type=int, help='Start batch number')
    parser.add_argument('--end', type=int, help='End batch number (inclusive)')
    parser.add_argument('--config', type=str, default='extreme_imbalance', help='Training config')
    parser.add_argument('--optuna-trials', type=int, default=15, help='Optuna trials per model')
    parser.add_argument('--timeout', type=int, default=2400, help='Training timeout per batch (s)')
    parser.add_argument('--disable-enhancements', action='store_true', help='Disable enhanced pipeline')
    parser.add_argument('--skip-collection', action='store_true', help='Skip data collection step if data exists')
    args = parser.parse_args()

    ensure_directories()

    t0 = time.time()
    successful: List[int] = []
    failed: List[int] = []

    # Determine batches to run
    batches: List[int] = []
    if args.batch is not None:
        batches = [args.batch]
    elif args.start is not None and args.end is not None:
        batches = list(range(args.start, args.end + 1))
    else:
        data = load_batch_data()
        batches = get_non_empty_batches(data)

    if not batches:
        logger.error("No batches to process")
        sys.exit(1)

    logger.info(f"Starting processing for batches: {batches}")
    for b in batches:
        ok = process_batch(
            batch_num=b,
            config=args.config,
            optuna_trials=args.optuna_trials,
            timeout=args.timeout,
            disable_enhancements=args.disable_enhancements,
            skip_collection=args.skip_collection,
        )
        (successful if ok else failed).append(b)

    total_time = time.time() - t0
    create_master_summary(successful, failed, total_time)
    logger.info(f"Done. Success: {successful} | Failed: {failed} | Time: {total_time:.1f}s")


if __name__ == '__main__':
    main()
