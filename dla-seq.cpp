#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <ctime>
#include <chrono>

using namespace std;

// Configuração padrão do DLA
int DIM = 500;
int SPAWN_DIST = 10;
double AGGREG_PROB = 1.0;
int PATH_LENGTH = 1000;
int NUM_PARTICLES = 2000;

// Configuração de cubos para visualização
const int CUBE_MIN_SIZE = 80;
const int CUBE_MAX_SIZE = 120;
const int CUBE_REPS = 20;

// Estrutura para representar as partículas
struct Point {
    int x, y, z;
    
    Point() : x(0), y(0), z(0) {}
    Point(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}
};

// Streams para construção do arquivo OBJ
stringstream faces_stream;
stringstream vertices_stream;

vector<int> grid;

void rotate(vector<float> &pV, vector<float> &pO, float theta, int d) {
    float a = pO[0];
    float b = pO[1];
    float c = pO[2];

    float x = pV[0];
    float y = pV[1];
    float z = pV[2];

    float u,v,w;

    if (d == 1) {u = 1;} else {u=0;}
    if (d == 2) {v = 1;} else {v=0;}
    if (d == 3) {w = 1;} else {w=0;}

    float x_new = (a * (v*v + w*w) - u * (b*v + c*w -u*x -v*y -w*z)) * (1 - cos(theta)) + x * cos(theta) + (-c*v + b*w - w*y + v*z) * sin(theta);
    float y_new = (b * (u*u + w*w) - v * (a*u + c*w -u*x -v*y -w*z)) * (1 - cos(theta)) + y * cos(theta) + (c*u - a*w + w*x - u*z) * sin(theta);
    float z_new = (c * (u*u + v*v) - w * (a*u + b*v -u*x -v*y -w*z)) * (1 - cos(theta)) + z * cos(theta) + (-b*u + a*v - v*x + u*y) * sin(theta);

    pV[0] = x_new;
    pV[1] = y_new;
    pV[2] = z_new;
}

void pointsToCubes(vector<Point>& points) {
    static int index = 0;

    for (const Point& point : points) {
        float x = static_cast<float>(point.x);
        float y = static_cast<float>(point.y);
        float z = static_cast<float>(point.z);

        float nn = (rand() % (CUBE_MAX_SIZE - CUBE_MIN_SIZE)) + CUBE_MIN_SIZE;
        float r = nn / 100.0f;
        int m = CUBE_REPS;
       
        for (int i = 0; i < m; i++) {
            vector<vector<float>> vertices = {
                {x-r, y+r, z+r}, {x+r, y+r, z+r}, {x+r, y-r, z+r}, {x-r, y-r, z+r},
                {x-r, y+r, z-r}, {x+r, y+r, z-r}, {x+r, y-r, z-r}, {x-r, y-r, z-r}
            };

            vector<vector<int>> faces = {
                {1,2,3}, {4,1,3}, {5,6,7}, {8,5,7},
                {1,5,2}, {6,2,5}, {4,3,8}, {7,8,3},
                {1,4,5}, {8,5,4}, {2,3,6}, {7,6,3}
            };

            float thetaX = static_cast<float>(rand() % 360);
            float thetaY = static_cast<float>(rand() % 360);
            float thetaZ = static_cast<float>(rand() % 360);

            for (vector<float>& v : vertices) {
                vector<float> center = {x, y, z};
                rotate(v, center, thetaX, 1);
                rotate(v, center, thetaY, 2);
                rotate(v, center, thetaZ, 3);

                v[0] = (v[0] - DIM / 2.0f) / DIM;
                v[1] = (v[1] - DIM / 2.0f) / DIM;
                v[2] = (v[2] - DIM / 2.0f) / DIM;

                vertices_stream << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
            }

            for (const vector<int>& f : faces) {
                faces_stream << "f";
                for (int vert_idx : f) {
                    faces_stream << " " << vert_idx + index;
                }
                faces_stream << "\n";
            }

            index += vertices.size();
        }
    }
}

// Função auxiliar para calcular índice no grid 3D
inline int getIndex(int x, int y, int z) {
    return z * DIM * DIM + y * DIM + x;
}

// Marca uma posição como frozen e atualiza docking points ao redor
void freezePoint(const Point& point, vector<Point>& frozen_points) {
    int idx = getIndex(point.x, point.y, point.z);
    grid[idx] = 2;  // 2 = frozen
    frozen_points.push_back(point);
    
    // Marca os 6 vizinhos como docking points (se não forem frozen)
    int dx[6] = {1, -1, 0, 0, 0, 0};
    int dy[6] = {0, 0, 1, -1, 0, 0};
    int dz[6] = {0, 0, 0, 0, 1, -1};
    
    for (int i = 0; i < 6; i++) {
        int nx = point.x + dx[i];
        int ny = point.y + dy[i];
        int nz = point.z + dz[i];
        
        if (nx >= 0 && nx < DIM && ny >= 0 && ny < DIM && nz >= 0 && nz < DIM) {
            int nidx = getIndex(nx, ny, nz);
            if (grid[nidx] == 0) {
                grid[nidx] = 1; 
            }
        }
    }
}

// Inicializa o grid com a semente
void seed(vector<Point>& frozen_points) {
    Point center(DIM/2, DIM/2, DIM/2);
    freezePoint(center, frozen_points);
}

bool tryAggregate() {
    double prob = static_cast<double>(rand()) / RAND_MAX;
    return prob <= AGGREG_PROB;
}

Point randomWalk(
    mt19937& rng,
    int spawning_dist,
    uniform_int_distribution<int>& step_dist,
    int path_length
) {
    uniform_int_distribution<int> spawn_box_dist(spawning_dist, DIM - spawning_dist - 1);
    uniform_int_distribution<int> side_dist(0, 5);

    Point current_pos;
    int side = side_dist(rng);
    if (side == 0) current_pos = Point(spawning_dist, spawn_box_dist(rng), spawn_box_dist(rng));
    if (side == 1) current_pos = Point(DIM - spawning_dist - 1, spawn_box_dist(rng), spawn_box_dist(rng));
    if (side == 2) current_pos = Point(spawn_box_dist(rng), spawning_dist, spawn_box_dist(rng));
    if (side == 3) current_pos = Point(spawn_box_dist(rng), DIM - spawning_dist - 1, spawn_box_dist(rng));
    if (side == 4) current_pos = Point(spawn_box_dist(rng), spawn_box_dist(rng), spawning_dist);
    if (side == 5) current_pos = Point(spawn_box_dist(rng), spawn_box_dist(rng), DIM - spawning_dist - 1);

    for (int step = 0; step < path_length; ++step) {
        // Move aleatoriamente em uma das 6 direções
        int dir = rand() % 6;
        if (dir == 0) current_pos.x++;
        else if (dir == 1) current_pos.x--;
        else if (dir == 2) current_pos.y++;
        else if (dir == 3) current_pos.y--;
        else if (dir == 4) current_pos.z++;
        else if (dir == 5) current_pos.z--;

        // Verifica limites
        if (current_pos.x < 0 || current_pos.x >= DIM ||
            current_pos.y < 0 || current_pos.y >= DIM ||
            current_pos.z < 0 || current_pos.z >= DIM) {
            return Point(-1, -1, -1);  // Saiu do grid
        }

        int idx = getIndex(current_pos.x, current_pos.y, current_pos.z);
        
        // Verifica se está em um docking point
        if (grid[idx] == 1) {
            if (tryAggregate()) {
                return current_pos;
            }
        }
    }

    return Point(-1, -1, -1);  // Não agregou
}

void printUsage(const char* program_name) {
    cout << "Uso: " << program_name << " <num_particulas> <dim_grid> <dist_spawn> <passos_caminho> <prob_agregacao> <arquivo_saida>" << endl;
    cout << "Exemplo: " << program_name << " 2000 500 10 1000 1.0 coral_cpu.obj" << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        NUM_PARTICLES = stoi(argv[1]);
        DIM = stoi(argv[2]);
        SPAWN_DIST = stoi(argv[3]);
        PATH_LENGTH = stoi(argv[4]);
        AGGREG_PROB = stod(argv[5]);
        string OUTFILE = argv[6];

        if (NUM_PARTICLES <= 0 || DIM <= 0 || SPAWN_DIST < 0 || PATH_LENGTH <= 0 || 
            AGGREG_PROB < 0.0 || AGGREG_PROB > 1.0) {
            cerr << "Erro: Parâmetros inválidos!" << endl;
            printUsage(argv[0]);
            return 1;
        }

        if (SPAWN_DIST >= DIM/2) {
            cerr << "Erro: SPAWN_DIST deve ser menor que DIM/2!" << endl;
            return 1;
        }

        // Inicializa o grid 3D
        size_t grid_size = (size_t)DIM * DIM * DIM;
        grid.resize(grid_size, 0);

        mt19937 rng(time(NULL));
        uniform_int_distribution<int> step_dist(-1, 1);

        vector<Point> frozen_points;
        seed(frozen_points);
        
        int spawning_dist = DIM / 2 - SPAWN_DIST;
        
        cout << "\nIniciando a simulação..." << endl;
        
        auto start = chrono::high_resolution_clock::now();
        
        while (frozen_points.size() < NUM_PARTICLES) {
            Point candidate = randomWalk(rng, spawning_dist, step_dist, PATH_LENGTH);

            if (candidate.x != -1) {
                freezePoint(candidate, frozen_points);
            }
        }

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        cout << "\nSimulação finalizada!" << endl;
        cout << "\nResultados" << endl;
        cout << "Tempo de execução: " << (duration.count() / 1000.0) << " segundos" << endl;
        cout << "Partículas congeladas: " << frozen_points.size() - 1 << endl;
        cout << "Total de pontos: " << frozen_points.size() << endl;

        cout << "\nGerando arquivo de saída..." << endl;

        pointsToCubes(frozen_points);

        ofstream out(OUTFILE);
        if (out.is_open()) {
            out << vertices_stream.str() << faces_stream.str();
            out.close();
            cout << "\nArquivo " << OUTFILE << " gerado!" << endl;
        }

    } catch (const exception& e) {
        cerr << "Erro ao processar argumentos: " << e.what() << endl;
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}