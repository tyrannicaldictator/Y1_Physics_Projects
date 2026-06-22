//Author: Minseo Kim (generated using GenAI, modified by Minseo Kim)
//Contains the code for the RK4 algorithm to numerically solve the N-body system

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

const double G = 6.67430e-11; 

struct SystemState {
    int N;
    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;

    SystemState(int n) : N(n), x(n, 0.0), y(n, 0.0), z(n, 0.0), 
                         vx(n, 0.0), vy(n, 0.0), vz(n, 0.0) {}
};

SystemState operator+(const SystemState& a, const SystemState& b) {
    SystemState res(a.N);
    for (int i = 0; i < a.N; ++i) {
        res.x[i] = a.x[i] + b.x[i];    res.y[i] = a.y[i] + b.y[i];    res.z[i] = a.z[i] + b.z[i];
        res.vx[i] = a.vx[i] + b.vx[i]; res.vy[i] = a.vy[i] + b.vy[i]; res.vz[i] = a.vz[i] + b.vz[i];
    }
    return res;
}

SystemState operator*(const SystemState& a, double scalar) {
    SystemState res(a.N);
    for (int i = 0; i < a.N; ++i) {
        res.x[i] = a.x[i] * scalar;    res.y[i] = a.y[i] * scalar;    res.z[i] = a.z[i] * scalar;
        res.vx[i] = a.vx[i] * scalar;  res.vy[i] = a.vy[i] * scalar;  res.vz[i] = a.vz[i] * scalar;
    }
    return res;
}

SystemState get_derivative(const SystemState& s, const std::vector<double>& masses) {
    SystemState deriv(s.N);

    for (int i = 0; i < s.N; ++i) {
        deriv.x[i] = s.vx[i];
        deriv.y[i] = s.vy[i];
        deriv.z[i] = s.vz[i];
    }

    for (int i = 0; i < s.N; ++i) {
        double ax = 0, ay = 0, az = 0;

        for (int j = 0; j < s.N; ++j) {
            if (i == j) continue;

            double dx = s.x[j] - s.x[i];
            double dy = s.y[j] - s.y[i];
            double dz = s.z[j] - s.z[i];
            
            double r2 = dx*dx + dy*dy + dz*dz;
            double r = std::sqrt(r2);
            double r3 = r2 * r;

            double a_mag = (G * masses[j]) / r3;
            
            ax += a_mag * dx;
            ay += a_mag * dy;
            az += a_mag * dz;
        }
        
        deriv.vx[i] = ax;
        deriv.vy[i] = ay;
        deriv.vz[i] = az;
    }
    return deriv;
}

SystemState rk4_step(const SystemState& s, const std::vector<double>& masses, double dt) {
    SystemState k1 = get_derivative(s, masses);
    SystemState k2 = get_derivative(s + k1 * (dt / 2.0), masses);
    SystemState k3 = get_derivative(s + k2 * (dt / 2.0), masses);
    SystemState k4 = get_derivative(s + k3 * dt, masses);
    
    return s + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
}

void shift_to_barycenter(SystemState& s, const std::vector<double>& masses) {
    double total_mass = 0;
    double cx = 0, cy = 0, cz = 0;
    double cvx = 0, cvy = 0, cvz = 0;

    for (int i = 0; i < s.N; ++i) {
        total_mass += masses[i];
        cx += masses[i] * s.x[i];       cy += masses[i] * s.y[i];       cz += masses[i] * s.z[i];
        cvx += masses[i] * s.vx[i];     cvy += masses[i] * s.vy[i];     cvz += masses[i] * s.vz[i];
    }

    cx /= total_mass; cy /= total_mass; cz /= total_mass;
    cvx /= total_mass; cvy /= total_mass; cvz /= total_mass;

    for (int i = 0; i < s.N; ++i) {
        s.x[i] -= cx;   s.y[i] -= cy;   s.z[i] -= cz;
        s.vx[i] -= cvx; s.vy[i] -= cvy; s.vz[i] -= cvz;
    }
}

int main() {
    int N = 9; 
    SystemState s(N);

    // Approximate masses (kg)
    std::vector<double> masses = {
        1.989e30,   
        3.301e23,   
        4.867e24,   
        5.972e24,   
        6.417e23,   
        1.898e27,   
        5.683e26,   
        8.681e25,   
        1.024e26    
    };

    // ---------------------------------------------------------
    // 3D Cartesian State Vectors (Approximate snapshot in time)
    // Positions in km
    // Velocities in km/s
    // Date: 2015/05/04
    // ---------------------------------------------------------
    

    std::vector<double> init_x = {
        -3.769839017281944e+05, -1.215887560476919e+07, -9.303802480249275e+07,
        -2.721905442823485e+07,  1.857829249673188e+08,  7.322290346218637e+08,
        -4.975105589537975e+08, -2.359308553370267e+09, -1.489383682227486e+09
    };


    std::vector<double> init_y = {
        -1.808551965070505e+05, -6.882140708223270e+07, -5.560364472640660e+07,
        -1.496852442684847e+08, -8.969495469221467e+07, 1.102444579356656e+08,
        1.257862036518430e+09, -1.441588394298658e+09, -4.279733733171386e+09
    };


    std::vector<double> init_z = {
        9.341587583473047e+03, -4.513561782662790e+06,  4.606080902036257e+06,
        9.561087398380041e+02, -6.450741952249847e+06, -1.685612115895072e+07,
        -2.231923766336977e+06,  2.526201033320862e+07,  1.224369012859621e+08
    };

    // Initial X velocities
    std::vector<double> init_vx = {
        4.166278235894496e-03,  3.824499092441582e+01,  1.773479641664427e+01,
        2.885152243025012e+01,  1.143213222274480e+01, -2.101893302123262e+00,
        -9.506106072633917e+00,  3.499893859704227e+00,  5.099573816895635e+00
    };

    // Initial Y velocities
    std::vector<double> init_vy = {
        -1.145464374099873e-02, -5.771007564516083e+00, -3.023049682816283e+01,
        -5.386519656425778e+00,  2.389387808846920e+01,  1.353158462463251e+01,
        -3.576941546699988e+00, -6.129273157686989e+00, -1.755512819209021e+00
    };

    // Initial Z velocities (Demonstrating motion into/out of the ecliptic)
    std::vector<double> init_vz = {
        -1.063592890504879e-04, -3.982781788981140e+00, -1.434753055097364e+00,
        6.462603221777385e-04,  2.190343478788233e-01, -8.628124565549733e-03,
        4.406016985135057e-01, -6.806150121929688e-02, -8.144111799087239e-02
    };

    for (int i = 0; i < N; ++i) {
        s.x[i] = init_x[i]*1000;     s.y[i] = init_y[i]*1000;     s.z[i] = init_z[i]*1000;
        s.vx[i] = init_vx[i]*1000;   s.vy[i] = init_vy[i]*1000;   s.vz[i] = init_vz[i]*1000;
    }
    shift_to_barycenter(s, masses);

    // Simulation Parameters
    double years = 50;
    double duration = 3600*24*365*years; 
    double dt = 3600; 
    int total_steps = duration / dt;

    std::ofstream outFile("solar_system_3d_cartesian_realfinal.csv");
    
    // Header
    outFile << "t";
    for(int i = 0; i < N; ++i) {
        outFile << ",x" << i << ",y" << i << ",z" << i;
    }
    outFile << "\n";

    double t = 0.0;
    for (int step = 0; step < total_steps; ++step) {
        outFile << t;
        for (int i = 0; i < N; ++i) {
            outFile << "," << s.x[i] << "," << s.y[i] << "," << s.z[i];
        }
        outFile << "\n";
        
        s = rk4_step(s, masses, dt);
        t += dt;
    }

    outFile.close();
    std::cout << "Successfully wrote 9-body 3D data to solar_system_3d_cartesian.csv\n";
    return 0;
}