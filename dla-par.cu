#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <ctime>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

// Configuração padrão do DLA
int DIM = 500;
int SPAWN_DIST = 10;
float AGGREG_PROB = 1.0f;
int PATH_LENGTH = 1000;
int NUM_PARTICLES = 2000;
int THREADS_PER_BLOCK = 256;

// Configuração de cubos para visualização
const int CUBE_MIN_SIZE = 80;
const int CUBE_MAX_SIZE = 120;
const int CUBE_REPS = 20;
const float PI = 3.1415926535f;

// Configuração de kernel
const int STEPS_PER_BATCH = 100;

// Streams para construção do arquivo OBJ
stringstream faces_stream;
stringstream vertices_stream;

// Estrutura para representar as partículas
struct Point {
    int x, y, z;
    int steps;
    bool active;
    
    __host__ __device__ Point() : x(0), y(0), z(0), steps(0), active(false) {}
    __host__ __device__ Point(int _x, int _y, int _z) : x(_x), y(_y), z(_z), steps(0), active(true) {}
};


__global__ void initRNG(curandState *states, unsigned long seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        curand_init(seed + tid * 137ULL, tid, 0, &states[tid]);
    }
}

// Kernel da caminhada aleatória, executado por cada partícula
__global__ void randomWalk(
    Point* particles,
    curandState* states,
    int* grid,
    Point* frozen_points,
    int* frozen_count,
    int dim,
    int spawn_distance,
    float aggreg_prob,
    int path_length,
    int num_particles,
    int target_frozen,
    int steps_per_batch
) {
    int start_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int end_tid = start_tid + 1;

    for (int tid = start_tid; tid < end_tid; ++tid) {
        if (tid >= num_particles) continue;
        
        curandState localState = states[tid];
        Point p = particles[tid];
        
        int respawn_counter = 0;
        int spawn_range = dim - 2 * spawn_distance;
        
        while (true) {
             // A verificação de parada global é feita apenas por threads "líderes" de warps
             // para reduzir a contenção. Em modo sequencial, tid é sempre 0, então a verificação ocorre.
            if (tid % 32 == 0) { 
                __threadfence();
                if (atomicAdd(frozen_count, 0) >= target_frozen) {
                    break;
                }
            }
            
            if (!p.active) {
                if (atomicAdd(frozen_count, 0) >= target_frozen) {
                    break;
                }
                
                int side = curand(&localState) % 6;
                
                switch(side) {
                    case 0:
                        p.x = spawn_distance;
                        p.y = spawn_distance + curand(&localState) % spawn_range;
                        p.z = spawn_distance + curand(&localState) % spawn_range;
                        break;
                    case 1:
                        p.x = dim - spawn_distance - 1;
                        p.y = spawn_distance + curand(&localState) % spawn_range;
                        p.z = spawn_distance + curand(&localState) % spawn_range;
                        break;
                    case 2:
                        p.y = spawn_distance;
                        p.x = spawn_distance + curand(&localState) % spawn_range;
                        p.z = spawn_distance + curand(&localState) % spawn_range;
                        break;
                    case 3:
                        p.y = dim - spawn_distance - 1;
                        p.x = spawn_distance + curand(&localState) % spawn_range;
                        p.z = spawn_distance + curand(&localState) % spawn_range;
                        break;
                    case 4:
                        p.z = spawn_distance;
                        p.x = spawn_distance + curand(&localState) % spawn_range;
                        p.y = spawn_distance + curand(&localState) % spawn_range;
                        break;
                    case 5:
                        p.z = dim - spawn_distance - 1;
                        p.x = spawn_distance + curand(&localState) % spawn_range;
                        p.y = spawn_distance + curand(&localState) % spawn_range;
                        break;
                }
                
                p.active = true;
                p.steps = 0;
                respawn_counter++;
            }
            
            for (int batch = 0; batch < steps_per_batch && p.active; batch++) {
                int dir = (int)(curand_uniform(&localState) * 6.0f);
                int dx = (dir == 0) - (dir == 1);
                int dy = (dir == 2) - (dir == 3);
                int dz = (dir == 4) - (dir == 5);
                
                p.x += dx;
                p.y += dy;
                p.z += dz;
                p.steps++;
                
                if (p.x < 0 || p.x >= dim || p.y < 0 || p.y >= dim || 
                    p.z < 0 || p.z >= dim || p.steps > path_length) {
                    p.active = false;
                    break;
                }
                
                int idx = p.z * dim * dim + p.y * dim + p.x;
                int current_val = grid[idx];
                
                bool near_docking = (current_val == 1);
                
                if (!near_docking && current_val == 0) {
                    int dx_arr[6] = {1, -1, 0, 0, 0, 0};
                    int dy_arr[6] = {0, 0, 1, -1, 0, 0};
                    int dz_arr[6] = {0, 0, 0, 0, 1, -1};
                    
                    for (int i = 0; i < 6 && !near_docking; i++) {
                        int nx = p.x + dx_arr[i];
                        int ny = p.y + dy_arr[i];
                        int nz = p.z + dz_arr[i];
                        
                        if (nx >= 0 && nx < dim && ny >= 0 && ny < dim && 
                            nz >= 0 && nz < dim) {
                            int nidx = nz * dim * dim + ny * dim + nx;
                            if (grid[nidx] == 1) {
                                near_docking = true;
                            }
                        }
                    }
                }
                
                if (near_docking && current_val == 1) {
                    float prob = curand_uniform(&localState);
                    if (prob <= aggreg_prob) {
                        int old = atomicCAS(&grid[idx], 1, 2);
                        
                        if (old == 1) {
                            int slot = atomicAdd(frozen_count, 1);
                            
                            if (slot < target_frozen) {
                                frozen_points[slot] = p;
                                
                                int dx_arr[6] = {1, -1, 0, 0, 0, 0};
                                int dy_arr[6] = {0, 0, 1, -1, 0, 0};
                                int dz_arr[6] = {0, 0, 0, 0, 1, -1};
                                
                                for (int i = 0; i < 6; i++) {
                                    int nx = p.x + dx_arr[i];
                                    int ny = p.y + dy_arr[i];
                                    int nz = p.z + dz_arr[i];
                                    
                                    if (nx >= 0 && nx < dim && ny >= 0 && ny < dim && 
                                        nz >= 0 && nz < dim) {
                                        int nidx = nz * dim * dim + ny * dim + nx;
                                        atomicCAS(&grid[nidx], 0, 1);
                                    }
                                }
                            }
                            
                            p.active = false;
                            break;
                        }
                    }
                }
            }
            
            __threadfence();
        }
        
        states[tid] = localState;
        particles[tid] = p;
    }
}


// Rotação dos cubos na hora de gerar o arquivo OBJ
void rotate(vector<float> &pV, vector<float> &pO, float theta, int d) {
    float a = pO[0], b = pO[1], c = pO[2];
    float x = pV[0], y = pV[1], z = pV[2];
    
    float u = (d == 1) ? 1.0f : 0.0f;
    float v = (d == 2) ? 1.0f : 0.0f;
    float w = (d == 3) ? 1.0f : 0.0f;
    
    float cos_t = cos(theta), sin_t = sin(theta);
    
    pV[0] = (a * (v*v + w*w) - u * (b*v + c*w - u*x - v*y - w*z)) * (1 - cos_t) + x * cos_t + (-c*v + b*w - w*y + v*z) * sin_t;
    pV[1] = (b * (u*u + w*w) - v * (a*u + c*w - u*x - v*y - w*z)) * (1 - cos_t) + y * cos_t + (c*u - a*w + w*x - u*z) * sin_t;
    pV[2] = (c * (u*u + v*v) - w * (a*u + b*v - u*x - v*y - w*z)) * (1 - cos_t) + z * cos_t + (-b*u + a*v - v*x + u*y) * sin_t;
}

// Geração dos cubos para o arquivo OBJ
void pointsToCubes(vector<Point>& points) {
    static int index = 0;
    for (const Point& point : points) {
        float x = (float)point.x, y = (float)point.y, z = (float)point.z;
        float r = ((rand() % (CUBE_MAX_SIZE - CUBE_MIN_SIZE)) + CUBE_MIN_SIZE) / 100.0f;
        
        for (int i = 0; i < CUBE_REPS; i++) {
            vector<vector<float>> vertices = {
                {x-r, y+r, z+r}, {x+r, y+r, z+r}, {x+r, y-r, z+r}, {x-r, y-r, z+r},
                {x-r, y+r, z-r}, {x+r, y+r, z-r}, {x+r, y-r, z-r}, {x-r, y-r, z-r}
            };
            
            float thetaX = (rand() % 360) * PI / 180.0f;
            float thetaY = (rand() % 360) * PI / 180.0f;
            float thetaZ = (rand() % 360) * PI / 180.0f;
            
            for (auto& v : vertices) {
                vector<float> center = {x, y, z};
                rotate(v, center, thetaX, 1);
                rotate(v, center, thetaY, 2);
                rotate(v, center, thetaZ, 3);
                
                v[0] = (v[0] - DIM / 2.0f) / DIM;
                v[1] = (v[1] - DIM / 2.0f) / DIM;
                v[2] = (v[2] - DIM / 2.0f) / DIM;
                
                vertices_stream << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
            }
            
            vector<vector<int>> faces = {
                {1,2,3}, {4,1,3}, {5,6,7}, {8,5,7}, {1,5,2}, {6,2,5},
                {4,3,8}, {7,8,3}, {1,4,5}, {8,5,4}, {2,3,6}, {7,6,3}
            };
            
            for (const auto& f : faces) {
                faces_stream << "f";
                for (int v : f) faces_stream << " " << v + index;
                faces_stream << "\n";
            }
            index += 8;
        }
    }
}


void printUsage(const char* prog) {
    cout << "Uso: " << prog << " <num_particulas> <dim_grid> <dist_spawn> <passos_caminho> <prob_agregacao> <threads_por_bloco> <arquivo_saida>" << endl;
    cout << "\nExemplo:" << endl;
    cout << "  Paralelo:   " << prog << " 2000 500 10 1000 1.0 256 coral.obj" << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 8) {
        printUsage(argv[0]);
        return 1;
    }
    
    try {
        NUM_PARTICLES = stoi(argv[1]);
        DIM = stoi(argv[2]);
        SPAWN_DIST = stoi(argv[3]);
        PATH_LENGTH = stoi(argv[4]);
        AGGREG_PROB = stof(argv[5]);
        THREADS_PER_BLOCK = stoi(argv[6]);
        string OUTFILE = argv[7];
        
        if (NUM_PARTICLES <= 0 || DIM <= 0 || SPAWN_DIST < 0 || 
            PATH_LENGTH <= 0 || AGGREG_PROB < 0.0 || AGGREG_PROB > 1.0 || 
            SPAWN_DIST >= DIM/2 || THREADS_PER_BLOCK <= 0 || THREADS_PER_BLOCK > 1024
        ) {
            cerr << "Parâmetros inválidos!" << endl;
            cerr << "Threads por bloco deve estar entre 1 e 1024." << endl;
            return 1;
        }

        srand(time(NULL));
        int spawning_dist = DIM / 2 - SPAWN_DIST;
        
        auto start = chrono::high_resolution_clock::now();

        // Alocações GPU
        Point *d_particles;
        curandState *d_states;
        int *d_grid, *d_frozen_count;
        Point *d_frozen_points;
        
        size_t grid_size = (size_t)DIM * DIM * DIM * sizeof(int);
        
        cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Point));
        cudaMalloc(&d_states, NUM_PARTICLES * sizeof(curandState));
        cudaMalloc(&d_grid, grid_size);
        cudaMalloc(&d_frozen_points, NUM_PARTICLES * sizeof(Point));
        cudaMalloc(&d_frozen_count, sizeof(int));

        cudaMemset(d_grid, 0, grid_size);
        cudaMemset(d_frozen_count, 0, sizeof(int));

        // Inicializa semente
        int center = DIM / 2;
        Point seed(center, center, center);
        vector<Point> h_frozen = {seed};
        
        int idx_seed = center * DIM * DIM + center * DIM + center;
        int val_frozen = 2;
        cudaMemcpy(d_grid + idx_seed, &val_frozen, sizeof(int), cudaMemcpyHostToDevice);
        
        int dx[6] = {1, -1, 0, 0, 0, 0};
        int dy[6] = {0, 0, 1, -1, 0, 0};
        int dz[6] = {0, 0, 0, 0, 1, -1};
        for (int i = 0; i < 6; i++) {
            int nx = center + dx[i];
            int ny = center + dy[i];
            int nz = center + dz[i];
            int nidx = nz * DIM * DIM + ny * DIM + nx;
            int val_dock = 1;
            cudaMemcpy(d_grid + nidx, &val_dock, sizeof(int), cudaMemcpyHostToDevice);
        }
        
        int threads_config = THREADS_PER_BLOCK;
        int blocks = (NUM_PARTICLES + threads_config - 1) / threads_config;      
            
        // A inicialização do RNG deve ser sempre paralela para inicializar todos os estados
        int init_blocks = (NUM_PARTICLES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        initRNG<<<init_blocks, THREADS_PER_BLOCK>>>(d_states, time(NULL), NUM_PARTICLES);
        cudaDeviceSynchronize();
        
        // Spawn inicial
        vector<Point> h_particles(NUM_PARTICLES);
        for (int i = 0; i < NUM_PARTICLES; i++) {
            int side = rand() % 6;
            int range = DIM - 2 * spawning_dist;
            Point p;
            switch(side) {
                case 0: p = Point(spawning_dist, spawning_dist + rand() % range, spawning_dist + rand() % range); break;
                case 1: p = Point(DIM - spawning_dist - 1, spawning_dist + rand() % range, spawning_dist + rand() % range); break;
                case 2: p = Point(spawning_dist + rand() % range, spawning_dist, spawning_dist + rand() % range); break;
                case 3: p = Point(spawning_dist + rand() % range, DIM - spawning_dist - 1, spawning_dist + rand() % range); break;
                case 4: p = Point(spawning_dist + rand() % range, spawning_dist + rand() % range, spawning_dist); break;
                case 5: p = Point(spawning_dist + rand() % range, spawning_dist + rand() % range, DIM - spawning_dist - 1); break;
            }
            h_particles[i] = p;
        }
        cudaMemcpy(d_particles, h_particles.data(), NUM_PARTICLES * sizeof(Point), cudaMemcpyHostToDevice);
        
        cout << "\nIniciando a simulação..." << endl;
        
        // Lança o kernel com a configuração de blocos e threads definida pelo modo
        randomWalk<<<blocks, threads_config>>>(
            d_particles,
            d_states,
            d_grid,
            d_frozen_points,
            d_frozen_count,
            DIM,
            spawning_dist,
            AGGREG_PROB,
            PATH_LENGTH,
            NUM_PARTICLES,
            NUM_PARTICLES,
            STEPS_PER_BATCH
        );
        
        // Espera as threads finalizarem
        cudaDeviceSynchronize();

        cout << "\nSimulação finalizada!" << endl;

        // Recupera partículas congeladas
        int final_count;
        cudaMemcpy(&final_count, d_frozen_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        final_count = min(final_count, NUM_PARTICLES);
        
        vector<Point> h_frozen_from_gpu(final_count);
        cudaMemcpy(h_frozen_from_gpu.data(), d_frozen_points, 
                   final_count * sizeof(Point), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < final_count; i++) {
            h_frozen.push_back(h_frozen_from_gpu[i]);
        }
        
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        cout << "\nResultados" << endl;
        cout << "Tempo de execução: " << (duration.count() / 1000.0) << " segundos" << endl;
        cout << "Partículas congeladas: " << final_count << endl;
        cout << "Total de pontos: " << h_frozen.size() << endl;

        cout << "\nGerando arquivo de saída..." << endl;
        
        pointsToCubes(h_frozen);
        
        ofstream out(OUTFILE);
        if (out.is_open()) {
            out << vertices_stream.str() << faces_stream.str();
            out.close();
            cout << "\nArquivo " << OUTFILE << " gerado!" << endl;
        }
        
        cudaFree(d_particles);
        cudaFree(d_states);
        cudaFree(d_grid);
        cudaFree(d_frozen_points);
        cudaFree(d_frozen_count);
        
    } catch (const exception& e) {
        cerr << "Erro: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}