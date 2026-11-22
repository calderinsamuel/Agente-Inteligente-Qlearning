"""Uso: python train.py --episodes 4000 --seed 42"""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.gridworld import GridWorld
from src.qlearning import train, evaluate, policy_text


def create_plots(training, path, output):
    fig, ax = plt.subplots(figsize=(9, 4))
    window = min(100, len(training.returns))
    series = np.convolve(training.returns, np.ones(window)/window, mode='valid')
    ax.plot(np.arange(len(series)) + window, series, label=f'Recompensa media ({window} episodios)', linewidth=1.8)
    ax.set(xlabel='Episodios', ylabel='Recompensa promedio', title='Evolución del entrenamiento Q-learning')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(output/'curva_aprendizaje.png', dpi=150);plt.close(fig)

    env = GridWorld()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_xlim(-.5, 7.5);ax.set_ylim(7.5, -.5)
    ax.set_xticks(np.arange(-.5, 8, 1));ax.set_yticks(np.arange(-.5, 8, 1))
    ax.grid(color='#d7dee8',linewidth=1)
    ax.set_xticklabels([]);ax.set_yticklabels([]);ax.tick_params(length=0)
    palette={'walls':'#243550','traps':'#f8b3b0','start':'#b9e8d0','goal':'#94d9f2'}
    from matplotlib.patches import Rectangle
    for r in range(8):
        for c in range(8):
            point=(r,c)
            fill=(palette['walls'] if point in env.walls else palette['traps'] if point in env.traps else
                  palette['start'] if point==env.start else palette['goal'] if point==env.goal else '#f8fafd')
            ax.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor=fill,edgecolor='none'))
            symbol=('S' if point==env.start else 'G' if point==env.goal else 'X' if point in env.traps else '■' if point in env.walls else '')
            if symbol: ax.text(c,r,symbol,ha='center',va='center',color='white' if point in env.walls else '#152741',weight='bold')
    for (r1,c1),(r2,c2) in zip(path[:-1],path[1:]):
        ax.annotate('',xy=(c2,r2),xytext=(c1,r1),arrowprops={'arrowstyle':'->','color':'#15569a','lw':2,'alpha':.9})
    ax.set_title('Ruta aprendida (S: inicio · G: meta · X: trampa)')
    fig.tight_layout(); fig.savefig(output/'ruta_aprendida.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description='Entrenamiento Q-learning para GridWorld')
    parser.add_argument('--episodes',type=int,default=4000)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    output=Path(__file__).parent/'outputs';output.mkdir(exist_ok=True)
    result=train(episodes=args.episodes,seed=args.seed)
    evaluation=evaluate(result.q_table)
    np.save(output/'q_table.npy',result.q_table)
    with (output/'entrenamiento.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file);writer.writerow(['episodio','recompensa','pasos','exito','epsilon'])
        for i,row in enumerate(zip(result.returns,result.lengths,result.success,result.epsilons),1):writer.writerow([i,*row])
    create_plots(result,evaluation['example_path'],output)
    summary={k:v for k,v in evaluation.items() if k!='example_path'}
    summary.update({'episodes':args.episodes,'seed':args.seed,'last_100_train_success_rate':float(np.mean(result.success[-100:])),
                    'path':[list(p) for p in evaluation['example_path']]})
    (output/'metricas.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Política aprendida:\n'+policy_text(result.q_table))
    print('\nMétricas:\n'+json.dumps(summary,indent=2,ensure_ascii=False))
    print('\nArchivos guardados en:',output)

if __name__=='__main__':main()
