from pathlib import Path
import argparse
import hashlib
import logging
from datetime import datetime, timedelta, timezone
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

MODELS = ['KPGT', 'KarmaDock', 'DrugCLIP']
METHODS = MODELS + ['RankAverage']
FRACTIONS = [(0.01, 'EF1%'), (0.05, 'EF5%'), (0.10, 'EF10%')]
COLORS = ['#2463a4', '#d58525', '#44966d', '#a84885']


def read_dataset(path):
    data = pd.read_csv(path, float_precision='round_trip')
    required = ['ID', 'SMILES', 'label'] + MODELS
    absent = set(required) - set(data.columns)
    if absent:
        raise ValueError(f'Missing columns: {sorted(absent)}')
    if data.ID.isna().any() or not data.ID.is_unique:
        raise ValueError('IDs must be nonempty and unique.')
    if data.SMILES.isna().any() or data.SMILES.astype(str).str.strip().eq('').any():
        raise ValueError('SMILES must be nonempty.')
    if data.label.isna().any() or set(data.label.unique()) != {0, 1}:
        raise ValueError('label must contain both 0 and 1, with no missing values.')
    for model in MODELS:
        data[model] = pd.to_numeric(data[model], errors='raise')
        if not np.isfinite(data[model]).all():
            raise ValueError(f'{model} contains missing or nonfinite scores.')
    if len(data) < 100:
        raise ValueError('At least 100 compounds are required for a nonzero top-1% cutoff.')
    return data


def enrichment_factor(labels, scores, fraction):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)
    k = int(np.floor(fraction * n))
    threshold = np.partition(scores, n-k)[n-k]
    above = scores > threshold
    tied = scores == threshold
    remaining = k - int(above.sum())
    hits = float(labels[above].sum()) + remaining * float(labels[tied].mean())
    ef = (hits/k) / (labels.sum()/n)
    return ef, k, hits


def calculate_metrics(data):
    ranks = data[MODELS].rank(ascending=False, method='average')
    mean_rank = ranks.mean(axis=1)
    priorities = {model: data[model].to_numpy() for model in MODELS}
    priorities['RankAverage'] = -mean_rank.to_numpy()
    labels = data.label.to_numpy(dtype=int)
    results, details = [], []
    for method in METHODS:
        score = priorities[method]
        row = {'Model': method}
        for fraction, metric in FRACTIONS:
            ef, k, hits = enrichment_factor(labels, score, fraction)
            row[metric] = ef
            details.append({'Model': method, 'cutoff': metric, 'selected': k,
                            'active_hits': hits})
        row['ROC-AUC'] = roc_auc_score(labels, score)
        results.append(row)
    return pd.DataFrame(results), priorities, pd.DataFrame(details)


def draw_figures(data, summary, priorities):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig_ef, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), layout='constrained')
    for ax, (fraction, metric) in zip(axes, FRACTIONS):
        values = summary[metric].to_numpy()
        bars = ax.bar(METHODS, values, color=COLORS, width=.65)
        ax.bar_label(bars, labels=[f'{value:.3f}' for value in values],
                     padding=4, fontsize=10)
        ax.axhline(1, color='gray', linestyle='--', linewidth=1)
        ax.set_title(metric)
        ax.set_ylabel('Enrichment factor')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_ylim(0, max(values)*1.17)
        ax.tick_params(axis='x', labelrotation=25)
    fig_ef.suptitle('Enrichment factors on the common scored set')

    fig_roc, ax = plt.subplots(figsize=(6.3, 5.5), layout='constrained')
    for method, color in zip(METHODS, COLORS):
        fpr, tpr, _ = roc_curve(data.label, priorities[method])
        auc = float(summary.loc[summary.Model == method, 'ROC-AUC'].iloc[0])
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f'{method} (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
            label='Random ranking')
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel='False positive rate',
           ylabel='True positive rate', title='ROC curves')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(loc='lower right', fontsize=9)
    return {'EF_bar_charts': fig_ef, 'ROC_curves': fig_roc}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path,
                        default=Path(__file__).resolve().with_name('merged_common_scores.csv'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent/'共同评分四指标结果',
                        help='Directory for calculation log, CSV and PNG/PDF; defaults to a results subfolder beside this script.')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not open interactive figure windows; useful for headless execution.')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('jak1_enrichment')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    for handler in [logging.FileHandler(args.output_dir/'calculation.log', mode='w', encoding='utf-8'),
                    logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    logger.info('JAK1 retrospective enrichment calculation')
    logger.info('Started (Asia/Shanghai): %s', datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'))
    logger.info('Input: %s', args.data.resolve())
    logger.info('Input SHA256: %s', hashlib.sha256(args.data.read_bytes()).hexdigest())
    logger.info('Script SHA256: %s', hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    logger.info('Python: %s', sys.version.split()[0])
    logger.info('Output directory: %s', args.output_dir.resolve())
    logger.info('Ranking: descending scores; average ties; consensus = equal-weight mean ranks.')
    logger.info('EF: k=floor(f*N); actual k/N denominator; boundary ties use expected active hits.')
    logger.info('Display/output precision: three decimals. Internal precision unchanged. No confidence intervals.')
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    data = read_dataset(args.data)
    logger.info('Validation passed: unique IDs, binary labels and finite scores for all three models.')
    logger.info('Compounds: %d; actives: %d; decoys: %d', len(data), int(data.label.sum()), int((data.label==0).sum()))
    summary, priorities, details = calculate_metrics(data)
    logger.info('\nMetrics summary\n%s', summary.to_string(index=False, float_format=lambda value: f'{value:.3f}'))
    logger.info('\nCalculation details (counts appear only in this log)\n%s',
                details.to_string(index=False, float_format=lambda value: f'{value:.3f}'))
    figures = draw_figures(data, summary, priorities)
    summary.to_csv(args.output_dir/'metrics_summary.csv', index=False,
                   encoding='utf-8-sig', float_format='%.3f')
    logger.info('Saved: metrics_summary.csv')
    for name, fig in figures.items():
        for extension in ['png', 'pdf']:
            fig.savefig(args.output_dir/f'{name}.{extension}', dpi=300)
            logger.info('Saved: %s.%s', name, extension)
    logger.info('Calculation and export completed successfully (Asia/Shanghai): %s',
                datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'))
    import matplotlib.pyplot as plt
    if not args.no_show:
        plt.show()
    for fig in figures.values():
        plt.close(fig)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.getLogger('jak1_enrichment').exception('Calculation failed.')
        raise