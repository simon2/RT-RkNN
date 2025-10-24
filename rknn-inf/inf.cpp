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
#include <unordered_map>
#include <algorithm>

#include <time.h>
#include <sys/time.h>

#include "inf.h"


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

        // Determine space boundaries (bounding box of all facilities and users)
        double min_x = numeric_limits<double>::max();
        double max_x = numeric_limits<double>::lowest();
        double min_y = numeric_limits<double>::max();
        double max_y = numeric_limits<double>::lowest();

        for (uint32_t i = 0; i < fac_cnt; ++i) {
            min_x = min(min_x, fac[i].x);
            max_x = max(max_x, fac[i].x);
            min_y = min(min_y, fac[i].y);
            max_y = max(max_y, fac[i].y);
        }
        for (uint32_t i = 0; i < usr_cnt; ++i) {
            min_x = min(min_x, usr[i].x);
            max_x = max(max_x, usr[i].x);
            min_y = min(min_y, usr[i].y);
            max_y = max(max_y, usr[i].y);
        }
        Rectangle space_boundaries(min_x, min_y, max_x, max_y);

        double filtering_time, verification_time;
        //
        // Filtering
        //
        cout << "Filtering..." << endl;
        start_time = get_wall_time();
        Point query_point(fac[q].x, fac[q].y, fac[q].id);
        vector<Line> bisectors;

        // Initialize vertex list with the four corners of space boundaries
        vector<Vertex> vertices;
        vertices.push_back(Vertex(Point(min_x, min_y), true));  // bottom-left
        vertices.push_back(Vertex(Point(max_x, min_y), true));  // bottom-right
        vertices.push_back(Vertex(Point(min_x, max_y), true));  // top-left
        vertices.push_back(Vertex(Point(max_x, max_y), true));  // top-right

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
                    Line new_bisector = perpendicular_bisector(query_point, current.point);

                    // Update vertex pruning counts for existing vertices
                    for (auto& vertex : vertices) {
                        if (!new_bisector.is_on_valid_side(vertex.point)) {
                            vertex.pruning_count++;
                        }
                    }

                    // Find intersections with space boundaries
                    vector<Point> boundary_intersections = find_line_rectangle_intersections(
                        new_bisector, space_boundaries);

                    // Add new intersection vertices
                    for (const auto& intersection : boundary_intersections) {
                        // Check if this intersection point already exists (within tolerance)
                        bool exists = false;
                        for (const auto& vertex : vertices) {
                            if (abs(vertex.point.x - intersection.x) < 1e-9 &&
                                abs(vertex.point.y - intersection.y) < 1e-9) {
                                exists = true;
                                break;
                            }
                        }

                        if (!exists) {
                            Vertex new_vertex(intersection, true);
                            // Count how many existing bisectors can prune this new vertex
                            for (const auto& bisector : bisectors) {
                                if (!bisector.is_on_valid_side(intersection)) {
                                    new_vertex.pruning_count++;
                                }
                            }
                            vertices.push_back(new_vertex);
                        }
                    }

                    // Find intersections with existing bisectors
                    for (const auto& existing_bisector : bisectors) {
                        Point intersection;
                        if (find_line_intersection(new_bisector, existing_bisector, intersection)) {
                            // Check if intersection is within space boundaries
                            if (intersection.x >= space_boundaries.min_x &&
                                intersection.x <= space_boundaries.max_x &&
                                intersection.y >= space_boundaries.min_y &&
                                intersection.y <= space_boundaries.max_y) {

                                // Check if this intersection point already exists
                                bool exists = false;
                                for (const auto& vertex : vertices) {
                                    if (abs(vertex.point.x - intersection.x) < 1e-9 &&
                                        abs(vertex.point.y - intersection.y) < 1e-9) {
                                        exists = true;
                                        break;
                                    }
                                }

                                if (!exists) {
                                    Vertex new_vertex(intersection, false);
                                    // Count how many bisectors can prune this new vertex
                                    for (const auto& bisector : bisectors) {
                                        if (!bisector.is_on_valid_side(intersection)) {
                                            new_vertex.pruning_count++;
                                        }
                                    }
                                    vertices.push_back(new_vertex);
                                }
                            }
                        }
                    }

                    bisectors.push_back(new_bisector);

                    // Remove vertices with pruning_count >= k (they cannot be part of the RkNN region)
                    vertices.erase(
                        remove_if(vertices.begin(), vertices.end(),
                            [k](const Vertex& v) {
                                return v.pruning_count >= (int)k;
                            }),
                        vertices.end()
                    );
                }
            } else {
                // This is a node - check if it can be pruned by existing bisectors
                bool can_prune = true;

                // Calculate minimum distance from current node to vertices with pruning_count = k-1
                // or extreme boundary points with pruning_count < k
                Rectangle node_mbr = current.node->get_mbr();

                // First, identify extreme points on each boundary edge
                // For each boundary edge, we need the two points with minimum and maximum coordinate along that edge
                Point* left_min = nullptr;   // Leftmost point with min y on left boundary
                Point* left_max = nullptr;   // Leftmost point with max y on left boundary
                Point* right_min = nullptr;  // Rightmost point with min y on right boundary
                Point* right_max = nullptr;  // Rightmost point with max y on right boundary
                Point* bottom_min = nullptr; // Bottom point with min x on bottom boundary
                Point* bottom_max = nullptr; // Bottom point with max x on bottom boundary
                Point* top_min = nullptr;    // Top point with min x on top boundary
                Point* top_max = nullptr;    // Top point with max x on top boundary

                const double eps = 1e-9;  // Tolerance for boundary checking

                // Find extreme points among boundary vertices with pruning_count < k
                for (auto& vertex : vertices) {
                    if (vertex.is_boundary && vertex.pruning_count < (int)k) {
                        // Check which boundary this vertex is on
                        if (abs(vertex.point.x - space_boundaries.min_x) < eps) {
                            // Left boundary
                            if (!left_min || vertex.point.y < left_min->y) {
                                left_min = &vertex.point;
                            }
                            if (!left_max || vertex.point.y > left_max->y) {
                                left_max = &vertex.point;
                            }
                        }
                        if (abs(vertex.point.x - space_boundaries.max_x) < eps) {
                            // Right boundary
                            if (!right_min || vertex.point.y < right_min->y) {
                                right_min = &vertex.point;
                            }
                            if (!right_max || vertex.point.y > right_max->y) {
                                right_max = &vertex.point;
                            }
                        }
                        if (abs(vertex.point.y - space_boundaries.min_y) < eps) {
                            // Bottom boundary
                            if (!bottom_min || vertex.point.x < bottom_min->x) {
                                bottom_min = &vertex.point;
                            }
                            if (!bottom_max || vertex.point.x > bottom_max->x) {
                                bottom_max = &vertex.point;
                            }
                        }
                        if (abs(vertex.point.y - space_boundaries.max_y) < eps) {
                            // Top boundary
                            if (!top_min || vertex.point.x < top_min->x) {
                                top_min = &vertex.point;
                            }
                            if (!top_max || vertex.point.x > top_max->x) {
                                top_max = &vertex.point;
                            }
                        }
                    }
                }

                for (const auto& vertex : vertices) {
                    bool should_consider = false;

                    // Condition 1: vertex with pruning_count == k-1
                    if (vertex.pruning_count == k - 1) {
                        should_consider = true;
                    }
                    // Condition 2: extreme boundary points with pruning_count < k
                    else if (vertex.is_boundary && vertex.pruning_count < (int)k) {
                        // Check if this vertex is one of the extreme points
                        if ((left_min && abs(vertex.point.x - left_min->x) < eps && abs(vertex.point.y - left_min->y) < eps) ||
                            (left_max && abs(vertex.point.x - left_max->x) < eps && abs(vertex.point.y - left_max->y) < eps) ||
                            (right_min && abs(vertex.point.x - right_min->x) < eps && abs(vertex.point.y - right_min->y) < eps) ||
                            (right_max && abs(vertex.point.x - right_max->x) < eps && abs(vertex.point.y - right_max->y) < eps) ||
                            (bottom_min && abs(vertex.point.x - bottom_min->x) < eps && abs(vertex.point.y - bottom_min->y) < eps) ||
                            (bottom_max && abs(vertex.point.x - bottom_max->x) < eps && abs(vertex.point.y - bottom_max->y) < eps) ||
                            (top_min && abs(vertex.point.x - top_min->x) < eps && abs(vertex.point.y - top_min->y) < eps) ||
                            (top_max && abs(vertex.point.x - top_max->x) < eps && abs(vertex.point.y - top_max->y) < eps)) {
                            should_consider = true;
                        }
                    }

                    if (should_consider &&
                        min_distance_to_rect(query_point, node_mbr) < query_point.distance_to(vertex.point)
                       ) {
                        can_prune = false;
                        break;
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

        // // Final vertex statistics
        end_time = get_wall_time();
        filtering_time = end_time - start_time;
        cout << "Filtering time: " << filtering_time << "[s]." << endl;
        cout << "Total bisectors created: " << bisectors.size() << endl << endl;

        //
        // Verification
        //
        cout << "Verifying..." << endl;
        start_time = get_wall_time();

        vector<Point> rknn_results;
        get_rknn_candidates(&usr_rtree, bisectors, rknn_results, k);

        end_time = get_wall_time();
        verification_time = end_time - start_time;
        cout << "Found " << rknn_results.size() << " RkNN results." << endl;
        cout << "Verification time: " << verification_time << "[s]." << endl << endl;

        cout << "Total time: " << filtering_time + verification_time << "[s]." << endl << endl;
    }
    catch( exception& e )
    {
        cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}