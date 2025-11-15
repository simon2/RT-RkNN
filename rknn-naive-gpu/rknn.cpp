#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <unordered_set>
#include <chrono>
#include <random>
#include <iomanip>

#include <time.h>
#include <sys/time.h>

// Include CUDA headers
#include <cuda_runtime.h>
#include "rknn_kernel.cuh"

using namespace std;

void printUsageAndExit( const char* argv0 )
{
    cerr << "Usage  : " << argv0 << " [options]\n";
    cerr << "Options: --infile | -if <filename>   Specify file for txt intput\n";
    cerr << "         --outfile | -of <filename>  Specify file for image output\n";
    cerr << "         --gpu                       Use GPU acceleration for RkNN computation\n";
    cerr << "         -k <number>                 Set k value for k-nearest neighbors (default: 4)\n";
    cerr << "         -q <number>                 Set query facility index (default: 0)\n";
    cerr << "         --help | -h                 Print this usage message\n";
    exit( 1 );
}

double get_wall_time()
{
  struct timeval time;
  if (gettimeofday(&time,NULL)){ 
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

// 2D Point class
class Point {
public:
    int x, y;
    int id;
    
    Point() : x(0), y(0), id(-1) {}
    Point(int x, int y, int id) : x(x), y(y), id(id) {}
    
    float distance_to(const Point& other) const {
        return sqrt(pow(x - other.x, 2) + pow(y - other.y, 2));
    }
    
    bool operator==(const Point& other) const {
        return id == other.id;
    }

    bool operator<(const Point& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return id < other.id;
    }
    
    friend ostream& operator<<(ostream& os, const Point& p) {
        os << "Point(" << p.x << ", " << p.y << ", id=" << p.id << ")";
        return os;
    }
};

// Naive spatial database without any indexing
class NaiveSpatialDB {
private:
    vector<Point> points;
    
public:
    void insert(const Point& point) {
        points.push_back(point);
    }
    
    void insert(int x, int y, int id) {
        points.emplace_back(x, y, id);
    }
    
    // Get all points
    const vector<Point>& get_all_points() const {
        return points;
    }
    
    // Find k nearest neighbors of a query point (brute force)
    vector<Point> knn_search(const Point& query, int k) const {
        vector<pair<float, Point>> distances;

        // Calculate distances to all points
        for (const auto& point : points) {
            if (point.id != query.id) {  // Exclude the query point itself
                float dist = query.distance_to(point);
                distances.emplace_back(dist, point);
            }
        }

        // Sort by distance
        sort(distances.begin(), distances.end());

        // Extract k nearest neighbors
        vector<Point> result;
        int limit = min(k, static_cast<int>(distances.size()));
        for (int i = 0; i < limit; ++i) {
            result.push_back(distances[i].second);
        }

        return result;
    }

    // Find k nearest neighbors including a specific external point (for reverse k-NN)
    vector<Point> knn_search_with_external(const Point& query, int k, const Point& external_point) const {
        vector<pair<float, Point>> distances;

        // Calculate distances to all points in database
        for (const auto& point : points) {
            if (point.id != query.id) {  // Exclude the query point itself
                float dist = query.distance_to(point);
                distances.emplace_back(dist, point);
            }
        }

        // Add the external point if it's different from query
        if (external_point.id != query.id) {
            float dist = query.distance_to(external_point);
            distances.emplace_back(dist, external_point);
        }

        // Sort by distance
        sort(distances.begin(), distances.end());

        // Extract k nearest neighbors
        vector<Point> result;
        int limit = min(k, static_cast<int>(distances.size()));
        for (int i = 0; i < limit; ++i) {
            result.push_back(distances[i].second);
        }

        return result;
    }
    
    // Range search within a given radius (brute force)
    vector<Point> range_search(const Point& center, float radius) const {
        vector<Point> result;
        
        for (const auto& point : points) {
            if (point.id != center.id && center.distance_to(point) <= radius) {
                result.push_back(point);
            }
        }
        
        return result;
    }
    
    size_t size() const {
        return points.size();
    }
};

// Naive Reverse k-NN implementation
class NaiveReverseKNN {
private:
    NaiveSpatialDB* user_db;
    NaiveSpatialDB* facility_db;
    bool use_gpu;

public:
    NaiveReverseKNN(NaiveSpatialDB* users, NaiveSpatialDB* facilities, bool gpu = false)
        : user_db(users), facility_db(facilities), use_gpu(gpu) {}

    // Method 1: CPU Brute Force Reverse k-NN
    // For each user, check if the query facility is among its k-NN facilities
    vector<Point> reverse_knn_bruteforce_cpu(const Point& query_facility, int k) const {
        vector<Point> result;
        const auto& all_users = user_db->get_all_points();
        const auto& all_facilities = facility_db->get_all_points();

        for (const auto& user : all_users) {
            // Find k-NN facilities for this user
            vector<pair<float, Point>> distances;

            // Calculate distances to all facilities
            for (const auto& facility : all_facilities) {
                float dist = user.distance_to(facility);
                distances.emplace_back(dist, facility);
            }

            // Sort by distance and get k nearest facilities
            sort(distances.begin(), distances.end());
            int limit = min(k, static_cast<int>(distances.size()));

            // Check if query facility is among the k nearest facilities
            for (int i = 0; i < limit; ++i) {
                if (distances[i].second.id == query_facility.id) {
                    result.push_back(user);
                    break;
                }
            }
        }

        return result;
    }

    // Method 2: GPU-accelerated Reverse k-NN
    vector<Point> reverse_knn_bruteforce_gpu(const Point& query_facility, int k) const {
        const auto& all_users = user_db->get_all_points();
        const auto& all_facilities = facility_db->get_all_points();

        // Convert Point objects to GPUPoint structures
        vector<GPUPoint> gpu_users(all_users.size());
        vector<GPUPoint> gpu_facilities(all_facilities.size());

        for (size_t i = 0; i < all_users.size(); ++i) {
            gpu_users[i].x = all_users[i].x;
            gpu_users[i].y = all_users[i].y;
            gpu_users[i].id = all_users[i].id;
        }

        for (size_t i = 0; i < all_facilities.size(); ++i) {
            gpu_facilities[i].x = all_facilities[i].x;
            gpu_facilities[i].y = all_facilities[i].y;
            gpu_facilities[i].id = all_facilities[i].id;
        }

        // Call GPU kernel wrapper
        vector<int> result_user_ids;
        gpuReverseKNN(gpu_users, gpu_facilities, query_facility.id, k, result_user_ids);

        // Convert result IDs back to Point objects
        vector<Point> result;
        for (int user_id : result_user_ids) {
            for (const auto& user : all_users) {
                if (user.id == user_id) {
                    result.push_back(user);
                    break;
                }
            }
        }

        return result;
    }

    // Reverse k-NN search using either CPU or GPU method
    vector<Point> reverse_knn(const Point& query, int k) const {
        if (use_gpu) {
            return reverse_knn_bruteforce_gpu(query, k);
        } else {
            return reverse_knn_bruteforce_cpu(query, k);
        }
    }

};

// Example usage and testing
int main( int argc, char* argv[] )
{
    string infile_path;
    string outfile;
    string algorithm_name = "all";
    uint32_t k = 4;
    uint32_t q = 0;
    bool use_gpu = false;

    for( int i = 1; i < argc; ++i )
    {
        const string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--infile" || arg == "-if" )
        {
            if( i < argc - 1 )
            {
                infile_path = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg == "--gpu" || arg == "-gpu" )
        {
            use_gpu = true;
            cout << "GPU acceleration enabled" << endl;
        }
        else if ( arg == "--algorithm" || arg == "-al" )
        {
            if( i < argc - 1 )
            {
                algorithm_name = argv[++i];
            }
        }
        else if ( arg == "-k" )
        {
            if (i < argc - 1 )
            {
                k = stoi(argv[++i]);
                if (k <= 0) 
                {
                    cerr << "Invalid value for k: " << k << ". It must be a positive integer." << endl;
                    exit(1);
                }
            }
            else
            {
                cerr << "Using default value k = 4." << endl;
            }
        }
        else if ( arg == "-q" )
        {
            if (i < argc - 1 )
            {
                q = stoi(argv[++i]);
                if (q < 0) 
                {
                    cerr << "Invalid value for q: " << q << ". It must be a >= 0 number." << endl;
                    printUsageAndExit( argv[0] );
                }
            }
            else
            {
                cerr << "Using default value q = 0." << endl;
            }
        }
        // else if( arg == "--outfile" || arg == "-of" )
        // {
        //     if( i < argc - 1 )
        //     {
        //         outfile = argv[++i];
        //     }
        //     else
        //     {
        //         printUsageAndExit( argv[0] );
        //     }
        // }
        else
        {
            cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        //
        // loading data
        //
        cout << "Loading data..." << endl;

        // open file
        ifstream infile(infile_path);
        if (!infile) 
        {
            cerr << "Error opening file: " << infile_path << endl;
            exit(1);
        }
        
        uint32_t fac_cnt;
        infile >> fac_cnt;

        Point *fac = new Point[fac_cnt];

        for (uint32_t i = 0; i < fac_cnt; ++i) 
        {
            infile >> fac[i].id;
            infile >> fac[i].x;
            infile >> fac[i].y;
        }

        uint32_t usr_cnt;
        infile >> usr_cnt;

        Point *usr = new Point[usr_cnt];

        for (uint32_t i = 0; i < usr_cnt; ++i) 
        {
            infile >> usr[i].id;
            infile >> usr[i].x;
            infile >> usr[i].y;
        }

        cout << "Loaded " << fac_cnt << " facilities and " << usr_cnt << " users." << endl;
        infile.close();

        // Create spatial database
        NaiveSpatialDB fac_db;
        for (uint32_t i = 0; i < fac_cnt; ++i) 
        {
            fac_db.insert(fac[i]);
        }

        NaiveSpatialDB usr_db;
        for (uint32_t i = 0; i < usr_cnt; ++i) 
        {
            usr_db.insert(usr[i]);
        }

        // Initialize CUDA if using GPU
        if (use_gpu) {
            int deviceCount;
            cudaGetDeviceCount(&deviceCount);
            if (deviceCount == 0) {
                cerr << "No CUDA-capable GPU device found. Falling back to CPU." << endl;
                use_gpu = false;
            } else {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, 0);
                cout << "Using GPU: " << prop.name << endl;
            }
        }

        NaiveReverseKNN rknn(&usr_db, &fac_db, use_gpu);

        double start_time = get_wall_time();
        vector<Point> rslts = rknn.reverse_knn(fac[q], k);
        double end_time = get_wall_time();
        cout << "RkNN search (" << (use_gpu ? "GPU" : "CPU") << ") completed in "
             << fixed << setprecision(6) << (end_time - start_time) << "[s].\n";
        cout << "Found " << rslts.size() << " RkNN results\n";
    }
    catch( exception& e )
    {
        cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}