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

using namespace std;

void printUsageAndExit( const char* argv0 )
{
    cerr << "Usage  : " << argv0 << " [options]\n";
    cerr << "Options: --infile | -if <filename>   Specify file for txt intput\n";
    cerr << "         --outfile | -of <filename>  Specify file for image output\n";
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
    double x, y;
    int id;
    
    Point() : x(0), y(0), id(-1) {}
    Point(double x, double y, int id) : x(x), y(y), id(id) {}
    
    double distance_to(const Point& other) const {
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
    
    void insert(double x, double y, int id) {
        points.emplace_back(x, y, id);
    }
    
    // Get all points
    const vector<Point>& get_all_points() const {
        return points;
    }
    
    // Find k nearest neighbors of a query point (brute force)
    vector<Point> knn_search(const Point& query, int k) const {
        vector<pair<double, Point>> distances;

        // Calculate distances to all points
        for (const auto& point : points) {
            if (point.id != query.id) {  // Exclude the query point itself
                double dist = query.distance_to(point);
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
        vector<pair<double, Point>> distances;

        // Calculate distances to all points in database
        for (const auto& point : points) {
            if (point.id != query.id) {  // Exclude the query point itself
                double dist = query.distance_to(point);
                distances.emplace_back(dist, point);
            }
        }

        // Add the external point if it's different from query
        if (external_point.id != query.id) {
            double dist = query.distance_to(external_point);
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
    vector<Point> range_search(const Point& center, double radius) const {
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
    NaiveSpatialDB* db;
    
public:
    NaiveReverseKNN(NaiveSpatialDB* database) : db(database) {}
    
    // Method 1: Brute Force Reverse k-NN
    // For each point in the database, check if the query point is among its k-NN
    vector<Point> reverse_knn_bruteforce(const Point& query, int k) const {
        vector<Point> result;
        const auto& all_points = db->get_all_points();

        for (const auto& point : all_points) {
            if (point.id == query.id) continue;  // Skip query point itself

            // Find k-NN of this point, including the query point as a potential neighbor
            auto knn = db->knn_search_with_external(point, k, query);

            // Check if query point is among the k-NN
            for (const auto& neighbor : knn) {
                if (neighbor.id == query.id) {
                    result.push_back(point);
                    break;
                }
            }
        }

        return result;
    }
    
    // Reverse k-NN search using brute force method
    vector<Point> reverse_knn(const Point& query, int k) const {
        return reverse_knn_bruteforce(query, k);
    }

    // Performance test method
    void test_reverse_knn(const Point& query, int k) const {
        cout << "\n=== Reverse " << k << "-NN for query point " << query << " ===\n";

        auto start = chrono::high_resolution_clock::now();
        auto result = reverse_knn_bruteforce(query, k);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

        cout << "Brute Force Method: " << result.size() << " results, "
                  << duration.count() << " μs\n";

        if (!result.empty()) {
            cout << "Results: ";
            for (const auto& point : result) {
                cout << "ID" << point.id << " ";
            }
            cout << "\n";
        }
    }
};

// Example usage and testing
int main( int argc, char* argv[] ) 
{
    string infile_path;
    string outfile;
    string algorithm_name = "all";
    uint32_t k;
    uint32_t q;

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
                cerr << "Missing value for k." << endl;
                printUsageAndExit( argv[0] );
            }
        }
        else if ( arg == "-q" )
        {
            if (i < argc - 1 )
            {
                q = stoi(argv[++i]);
                if (q <= 0) 
                {
                    cerr << "Invalid value for q: " << q << ". It must be a positive number." << endl;
                    printUsageAndExit( argv[0] );
                }
            }
            else
            {
                cerr << "Missing value for q." << endl;
                printUsageAndExit( argv[0] );
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

        double start_time, end_time;

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
        NaiveReverseKNN rknn(&usr_db);

        vector<Point> rslts = rknn.reverse_knn(fac[q], k);
        cout << "got " << rslts.size() << " results\n";
    }
    catch( exception& e )
    {
        cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    // cout << "=== Naive Reverse k-NN Implementation ===\n\n";
    
    // // Create spatial database
    // NaiveSpatialDB db;
    // NaiveReverseKNN rknn(&db);
    
    // // Test with small dataset first
    // cout << "--- Small Test Dataset ---\n";
    
    // // Insert some test points
    // db.insert(1.0, 1.0, 1);
    // db.insert(2.0, 2.0, 2);
    // db.insert(3.0, 1.0, 3);
    // db.insert(1.0, 3.0, 4);
    // db.insert(4.0, 4.0, 5);
    // db.insert(5.0, 2.0, 6);
    // db.insert(2.0, 5.0, 7);
    // db.insert(6.0, 1.0, 8);
    // db.insert(1.5, 4.5, 9);
    // db.insert(3.5, 3.5, 10);
    
    // cout << "Database size: " << db.size() << " points\n";
    
    // // Test queries
    // Point query1(2.5, 2.5, 100);  // Query point not in database
    // Point query2(3.0, 1.0, 3);    // Query point in database
    
    // rknn.test_reverse_knn(query1, 2);
    // rknn.test_reverse_knn(query1, 3);
    // rknn.test_reverse_knn(query2, 2);
    
    // // Test with larger random dataset
    // cout << "\n--- Larger Random Dataset ---\n";
    
    // NaiveSpatialDB large_db;
    // NaiveReverseKNN large_rknn(&large_db);
    
    // // Generate random points
    // auto random_points = generate_random_points(50, 0.0, 20.0);
    // for (const auto& point : random_points) {
    //     large_db.insert(point);
    // }
    
    // cout << "Database size: " << large_db.size() << " points\n";
    
    // // Test with random query points
    // Point large_query1(10.0, 10.0, 999);
    // Point large_query2 = random_points[25];  // Use an existing point
    
    // large_rknn.test_reverse_knn(large_query1, 3);
    // large_rknn.test_reverse_knn(large_query1, 5);
    // large_rknn.test_reverse_knn(large_query2, 3);
    
    // // Demonstrate what reverse k-NN means
    // cout << "\n--- Understanding Reverse k-NN ---\n";
    // cout << "Query point: " << large_query1 << "\n";
    // cout << "Finding points that have the query as one of their 3-NN:\n\n";
    
    // auto rknn_result = large_rknn.reverse_knn_bruteforce(large_query1, 3);
    
    // for (const auto& point : rknn_result) {
    //     auto knn = large_db.knn_search(point, 3);
    //     cout << "Point " << point << " has 3-NN: ";
    //     for (const auto& neighbor : knn) {
    //         cout << "ID" << neighbor.id << "(d=" << fixed << setprecision(2) 
    //                   << point.distance_to(neighbor) << ") ";
    //     }
    //     cout << "\n";
    // }
    
    return 0;
}