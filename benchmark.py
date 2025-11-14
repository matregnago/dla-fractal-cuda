import subprocess
import os
import re
import statistics
import csv

# Configurações gerais
EXECUTABLE_CPP = "./dla_seq"
EXECUTABLE_CUDA = "./dla_par"
SALVAR_ARQUIVOS = False # Salvar os arquivos .obj
NUM_RUNS = 1 # Número de vezes que cada teste será executado
OUTPUT_CSV_FILE = "resultados.csv"

# Testes a serem executados
tests = [
    { 'name': 'Escala_Particulas_10k', 'params': { 'num_particles': 10000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_20k', 'params': { 'num_particles': 20000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_30k', 'params': { 'num_particles': 30000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_40k', 'params': { 'num_particles': 40000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_50k', 'params': { 'num_particles': 50000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_60k', 'params': { 'num_particles': 60000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_70k', 'params': { 'num_particles': 70000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_80k', 'params': { 'num_particles': 80000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_90k', 'params': { 'num_particles': 90000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_100k', 'params': { 'num_particles': 100000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_110k', 'params': { 'num_particles': 110000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_120k', 'params': { 'num_particles': 120000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_130k', 'params': { 'num_particles': 130000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_140k', 'params': { 'num_particles': 140000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_150k', 'params': { 'num_particles': 150000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_160k', 'params': { 'num_particles': 160000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_170k', 'params': { 'num_particles': 170000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_180k', 'params': { 'num_particles': 180000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_190k', 'params': { 'num_particles': 190000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
    { 'name': 'Escala_Particulas_200k', 'params': { 'num_particles': 200000, 'dim_grid': 700, 'spawn_dist': 90, 'path_length': 80000, 'prob_aggreg': 1.0, 'threads_per_block': 256 } },
]


def run_and_get_time(command):
    try:
        print(f"Executando: {' '.join(command)}")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )

        match = re.search(r"Tempo de execução: (\d+\.?\d*)", result.stdout)
        
        if match:
            return float(match.group(1))
        else:
            print("Não foi possível encontrar o tempo de execução na saída do programa.")
            print("Saída recebida:\n" + result.stdout)
            return None

    except FileNotFoundError:
        print(f"ERRO: Executável não encontrado: {command[0]}")
        print("Verifique se o programa foi compilado e se o nome no script está correto.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"ERRO: O programa falhou ao executar (código de saída {e.returncode}).")
        print(f"Comando: {' '.join(e.cmd)}")
        print(f"Erro reportado:\n{e.stderr}")
        return None


def main():
    all_results = []
    
    print("Iniciando testes...")

    for test in tests:
        test_name = test['name']
        p = test['params']
        print(f"Executando: {test_name}")

        times_cpu_seq = []
        times_gpu_par = []

        # Execução do sequencial
        print(f"1. Testando sequencial ({NUM_RUNS} execuções)")
        for i in range(NUM_RUNS):
            output_file = f"output_{test_name.replace(' ', '_').lower()}_cpu.obj"
            command = [
                EXECUTABLE_CPP,
                str(p['num_particles']), str(p['dim_grid']), str(p['spawn_dist']),
                str(p['path_length']), str(p['prob_aggreg']), output_file
            ]
            time = run_and_get_time(command)
            if time is not None:
                times_cpu_seq.append(time)
            if not SALVAR_ARQUIVOS and os.path.exists(output_file):
                os.remove(output_file)
                
        # Execução do paralelo na GPU
        print(f"\n2. Testando paralelo na GPU ({NUM_RUNS} execuções)")
        for i in range(NUM_RUNS):
            output_file = f"output_{test_name.replace(' ', '_').lower()}_gpu_par.obj"
            command = [
                EXECUTABLE_CUDA,
                str(p['num_particles']), str(p['dim_grid']), str(p['spawn_dist']),
                str(p['path_length']), str(p['prob_aggreg']), str(p['threads_per_block']),
                output_file
            ]
            time = run_and_get_time(command)
            if time is not None:
                times_gpu_par.append(time)
            if not SALVAR_ARQUIVOS and os.path.exists(output_file):
                os.remove(output_file)

        # Calcula as médias
        avg_cpu_seq = statistics.mean(times_cpu_seq) if times_cpu_seq else 0.0
        avg_gpu_par = statistics.mean(times_gpu_par) if times_gpu_par else 0.0

        all_results.append({
            'Nome do teste': test_name,
            'Média do tempo sequencial': f"{avg_cpu_seq:.4f}",
            'Média do tempo paralelo na GPU': f"{avg_gpu_par:.4f}",
        })
        
        print(f"\nResultados para '{test_name}':")
        print(f"  - Média sequencial: {avg_cpu_seq:.4f}s")
        print(f"  - Média paralelo na GPU:   {avg_gpu_par:.4f}s")

    print(f"\nGerando arquivo de resultados: {OUTPUT_CSV_FILE}")
    
    header = [
        'Nome do teste', 
        'Média do tempo sequencial', 
        'Média do tempo paralelo na GPU'
    ]
    
    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_results)
    
    print("Arquivo CSV gerado com sucesso!")
        
if __name__ == "__main__":
    main()
