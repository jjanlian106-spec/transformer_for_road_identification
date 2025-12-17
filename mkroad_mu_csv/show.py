import argparse
from pathlib import Path
import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool


def read_csv_safe(path: Path):
    df = pd.read_csv(path)
    # ensure index column exists
    if 'index' not in df.columns:
        raise ValueError(f"CSV {path} must contain column 'index'")
    # normalize
    df['index'] = pd.to_numeric(df['index'], errors='coerce')
    df['mu'] = pd.to_numeric(df['mu'], errors='coerce')
    # keep path column if exists
    if 'path' not in df.columns:
        df['path'] = ''
    return df


def main():
    parser = argparse.ArgumentParser()
    fusi_path = "road_info/fusion_road_info/fusion_road_mu.csv"
    pred_path = "road_info/predict_road_info/predict_road_mu.csv"
    real_path = "road_info/real_road_info/real_road_mu.csv"
    dynamic_path = "road_info/dynamic_road_info/dynamic_road_mu.csv"
    parser.add_argument('--fusi-csv', type=str, default=fusi_path)
    parser.add_argument('--pred-csv', type=str, default=pred_path)
    parser.add_argument('--real-csv', type=str, default=real_path)
    parser.add_argument('--dynamic-csv', type=str, default=dynamic_path)
    parser.add_argument('--out-html', type=str, default='road_info/mu_result.html')
    args = parser.parse_args()

    fusi_path = Path(args.fusi_csv)
    pred_path = Path(args.pred_csv)
    real_path = Path(args.real_csv)
    dynamic_path = Path(args.dynamic_csv)

    if not pred_path.exists():
        raise SystemExit(f'Predicted CSV not found: {pred_path}')
    if not real_path.exists():
        raise SystemExit(f'Real CSV not found: {real_path}')
    if not fusi_path.exists():
        raise SystemExit(f'Dynamic estimated CSV not found: {fusi_path}')
    if not dynamic_path.exists():
        raise SystemExit(f'Dynamic estimated CSV not found: {dynamic_path}')

    df_fusi = read_csv_safe(fusi_path)
    df_pred = read_csv_safe(pred_path)
    df_real = read_csv_safe(real_path)
    df_dynamic = read_csv_safe(dynamic_path)

    # rename mu and path columns to distinguish
    df_fusi = df_fusi.rename(columns={'mu': 'mu_fusi', 'path': 'path_fusi'})
    df_pred = df_pred.rename(columns={'mu': 'mu_pred', 'path': 'path_pred'})
    df_real = df_real.rename(columns={'mu': 'mu_real', 'path': 'path_real'})
    df_dynamic = df_dynamic.rename(columns={'mu': 'mu_dynamic', 'path': 'path_dynamic'})

    # merge on index using chain merge
    df = pd.merge(df_fusi[['index', 'mu_fusi', 'path_fusi']],
                  df_pred[['index', 'mu_pred', 'path_pred']],
                  on='index', how='outer')
    df = pd.merge(df, df_real[['index', 'mu_real', 'path_real']], on='index', how='outer')
    df = pd.merge(df, df_dynamic[['index', 'mu_dynamic', 'path_dynamic']], on='index', how='outer')
    df = df.sort_values('index')
    df = df.fillna({'mu_fusi': float('nan'),'mu_pred': float('nan'), 'mu_real': float('nan'),'mu_dynamic': float('nan'),'path_fusi':'','path_pred': '', 'path_real': '','path_dynamic': ''})

    # Convert index to time by multiplying by 0.001
    df['index'] = df['index'] * 0.001

    source = ColumnDataSource(df)

    # wider canvas and responsive width to avoid narrow display
    p = figure(title='Predicted vs Real mu', x_axis_label='time (s)', y_axis_label='mu',
               tools='pan,wheel_zoom,box_zoom,reset,save',
               width=1400, height=800, sizing_mode='stretch_width')
    p.line('index', 'mu_fusi', source=source, color='green', legend_label='fusion mu', line_width=2)
    p.line('index', 'mu_real', source=source, color='red', legend_label='real mu', line_width=2)
    p.line('index', 'mu_pred', source=source, color='blue', legend_label='vison mu', line_width=2)
    p.line('index', 'mu_dynamic', source=source, color='orange', legend_label='dynamic mu', line_width=2)
    hover = HoverTool(tooltips=[('time', '@index'),
                                ('fusi mu', '@mu_fusi'),
                                ('pred mu', '@mu_pred'),
                                ('real mu', '@mu_real'),
                                ('dynamic mu', '@mu_dynamic'),
                                ('fusi path', '@path_fusi'),
                                ('pred path', '@path_pred'),
                                ('real path', '@path_real'),
                                ('dynamic path', '@path_dynamic')])
    p.add_tools(hover)
    p.legend.location = 'top_left'
    p.legend.click_policy = 'hide'

    out_html = args.out_html
    output_file(out_html)
    save(p)
    print(f'Wrote plot to {out_html}')


if __name__ == '__main__':
    main()
