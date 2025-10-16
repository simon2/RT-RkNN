#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <queue>
#include <stack>
#include <vector>
#include <limits>
#include <random>

#include <time.h>
#include <sys/time.h>

#include "tpl.h"

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

int main( int argc, char* argv[] ) 
{
    string infile_path;
    string outfile;
    string algorithm_name = "all";
    uint32_t k = 4;
    uint32_t q = 0;

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

        double start_time, end_time;

        // Build R*-Tree
        cout << "building R*-trees..." << endl;
        start_time = get_wall_time();
        RStarTree fac_rtree;
        RStarTree usr_rtree;
        for (uint32_t i = 0; i < fac_cnt; ++i) 
        {
            fac_rtree.insert(fac[i]);
        }
        for (uint32_t i = 0; i < usr_cnt; ++i) 
        {
            usr_rtree.insert(usr[i]);
        }
        end_time = get_wall_time();
        cout << "R*-tree is built in " << end_time - start_time << "[s]." <<  endl << endl;

        //
        // Filtering
        //
        cout << "Filtering..." << endl;
        start_time = get_wall_time();
        Point query_point(fac[q].x, fac[q].y, fac[q].id);
        vector<Line> bisectors;

        // Priority queue-based bisector creation
        // Use min-heap priority queue (closest entries first)
        priority_queue<PQEntry, vector<PQEntry>, greater<PQEntry>> pq;

        // Initialize with root node
        auto root = fac_rtree.get_root();
        if (root) {
            PQEntry root_entry;
            root_entry.node = root;
            root_entry.distance = min_distance_to_rect(query_point, root->get_mbr());
            root_entry.is_point = false;
            pq.push(root_entry);
        }

        // Process entries from priority queue
        while (!pq.empty()) {
            PQEntry current = pq.top();
            pq.pop();

            if (current.is_point) {
                // This is a point - create a bisector if it's not the query point itself
                if (current.point.id != query_point.id) {
                    bisectors.push_back(perpendicular_bisector(query_point, current.point));
                }
            } else {
                // This is a node - check if it can be pruned by existing bisectors
                bool can_prune = false;
                int violation_count = 0;

                for (const auto& bisector : bisectors) {
                    if (can_prune_node_by_valid_side(current.node, bisector)) {
                        violation_count++;
                        if (violation_count > (int)k) {
                            can_prune = true;
                            break;
                        }
                    }
                }

                // If not pruned, add its children/points to the priority queue
                if (!can_prune) {
                    if (current.node->is_leaf) {
                        // Add points from leaf node
                        for (const auto& entry : current.node->entries) {
                            PQEntry point_entry;
                            point_entry.is_point = true;
                            point_entry.point = entry.point;
                            point_entry.distance = query_point.distance_to(entry.point);
                            point_entry.node = nullptr;
                            pq.push(point_entry);
                        }
                    } else {
                        // Add child nodes from internal node
                        for (const auto& entry : current.node->entries) {
                            if (entry.child) {
                                PQEntry child_entry;
                                child_entry.node = entry.child;
                                child_entry.distance = min_distance_to_rect(query_point, entry.child->get_mbr());
                                child_entry.is_point = false;
                                pq.push(child_entry);
                            }
                        }
                    }
                }
            }
        }
        vector<Point> rknn_candidates;
        get_rknn_candidates(&usr_rtree, bisectors, rknn_candidates, k);

        end_time = get_wall_time();
        double filtering_time = end_time - start_time;
        cout << "Found " << rknn_candidates.size() << " RkNN candidates." << endl;
        cout << "Filtering time: " << filtering_time << "[s]." << endl << endl;

        //
        // Verification
        //
        cout << "Verifying..." << endl;
        start_time = get_wall_time();
        vector<Point> final_rknn;
        for (const auto& candidate : rknn_candidates) {
            auto fac_knn_of_candidate = fac_rtree.knn_search(candidate, k);
            bool found = false;
            for (const auto& neighbor : fac_knn_of_candidate) {
                if (query_point.id == neighbor.id) {
                    found = true;
                    break;
                }
            }
            if (found) {
                final_rknn.push_back(candidate);
            }
        }
        end_time = get_wall_time();
        double verification_time = end_time - start_time;
        cout << "Verification time: " << verification_time << "[s]." << endl << endl;
        cout << "Found " << final_rknn.size() << " RkNN results." << endl;
        
        cout << "Total time: " << filtering_time + verification_time << "[s]." << endl << endl;
    }
    catch( exception& e )
    {
        cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}