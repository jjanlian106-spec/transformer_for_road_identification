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
    pred_path = "road_info/predict_road_info/predict_road_mu.csv"
    real_path = "road_info/real_road_info/real_road_mu.csv"
    parser.add_argument('--pred-csv', type=str, default=pred_path)
    parser.add_argument('--real-csv', type=str, default=real_path)
    parser.add_argument('--out-html', type=str, default='road_info/predict_vs_real_mu.html')
    args = parser.parse_args()

    pred_path = Path(args.pred_csv)
    real_path = Path(args.real_csv)

    if not pred_path.exists():
        raise SystemExit(f'Predicted CSV not found: {pred_path}')
    if not real_path.exists():
        raise SystemExit(f'Real CSV not found: {real_path}')

    df_pred = read_csv_safe(pred_path)
    df_real = read_csv_safe(real_path)

    # rename mu and path columns to distinguish
    df_pred = df_pred.rename(columns={'mu': 'mu_pred', 'path': 'path_pred'})
    df_real = df_real.rename(columns={'mu': 'mu_real', 'path': 'path_real'})

    # merge on index
    df = pd.merge(df_pred[['index', 'mu_pred', 'path_pred']],
                  df_real[['index', 'mu_real', 'path_real']],
                  on='index', how='outer')
    df = df.sort_values('index')
    df = df.fillna({'mu_pred': float('nan'), 'mu_real': float('nan'), 'path_pred': '', 'path_real': ''})

    source = ColumnDataSource(df)

    # wider canvas and responsive width to avoid narrow display
    p = figure(title='Predicted vs Real mu', x_axis_label='index', y_axis_label='mu',
               tools='pan,wheel_zoom,box_zoom,reset,save',
               width=1400, height=800, sizing_mode='stretch_width')
    p.line('index', 'mu_real', source=source, color='red', legend_label='real mu', line_width=2)
    p.line('index', 'mu_pred', source=source, color='blue', legend_label='predicted mu', line_width=2)
    hover = HoverTool(tooltips=[('index', '@index'),
                                ('pred mu', '@mu_pred'),
                                ('real mu', '@mu_real'),
                                ('pred path', '@path_pred'),
                                ('real path', '@path_real')])
    p.add_tools(hover)
    p.legend.location = 'top_left'

    out_html = args.out_html
    output_file(out_html)
    save(p)
    print(f'Wrote plot to {out_html}')


if __name__ == '__main__':
    main()
