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

#include "RStarTree2D.h"

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

int main( int argc, char* argv[] ) {
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

        cout << "R*-tree is built in " << end_time - start_time << endl << endl;

        // K-nearest neighbors search
        // int k = 3;
        cout << "Finding " << k*2 << " nearest neighbors to point ("
                << fac[q].x << ", " << fac[q].y << "):" << endl;
        start_time = get_wall_time();
        auto knn_results = usr_rtree.knn_search(fac[q], k*2);
        end_time = get_wall_time();
        cout << "KNN search time: " << end_time - start_time << "[s]." << endl << endl;
        for (size_t i = 0; i < knn_results.size(); ++i) {
            const auto& point = knn_results[i];
            std::cout << (i + 1) << ". Point: (" << point.x << ", " << point.y
                    << ") ID: " << point.id
                    << " Distance: " << fac[q].distance_to(point) << std::endl;
        }
        cout << endl;
    }
    catch( exception& e )
    {
        cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }


    // // Range search
    // std::cout << "Range search for points in rectangle (2.0, 2.0) to (5.0, 5.0):" << std::endl;
    // auto range_results = rtree.range_search(2.0, 2.0, 5.0, 5.0);
    // for (const auto& point : range_results) {
    //     std::cout << "Found point: (" << point.x << ", " << point.y
    //               << ") ID: " << point.id << std::endl;
    // }
    // std::cout << std::endl;

    // // Radius search
    // Point query_point(3.0, 3.0, -1);
    // double radius = 2.0;
    // std::cout << "Radius search around point (" << query_point.x << ", " << query_point.y
    //           << ") with radius " << radius << ":" << std::endl;
    // auto radius_results = rtree.radius_search(query_point, radius);
    // for (const auto& point : radius_results) {
    //     std::cout << "Found point: (" << point.x << ", " << point.y
    //               << ") ID: " << point.id
    //               << " Distance: " << query_point.distance_to(point) << std::endl;
    // }
    // std::cout << std::endl;

    // // K-nearest neighbors search
    // int k = 3;
    // std::cout << "Finding " << k << " nearest neighbors to point ("
    //           << query_point.x << ", " << query_point.y << "):" << std::endl;
    // auto knn_results = rtree.knn_search(query_point, k);
    // for (size_t i = 0; i < knn_results.size(); ++i) {
    //     const auto& point = knn_results[i];
    //     std::cout << (i + 1) << ". Point: (" << point.x << ", " << point.y
    //               << ") ID: " << point.id
    //               << " Distance: " << query_point.distance_to(point) << std::endl;
    // }
    // std::cout << std::endl;

    // // Remove a point
    // std::cout << "Removing point (2.5, 2.5) with ID 2..." << std::endl;
    // bool removed = rtree.remove(2.5, 2.5, 2);
    // std::cout << "Point " << (removed ? "successfully removed" : "not found") << std::endl;

    // // Verify removal with another range search
    // std::cout << "\nRange search after removal:" << std::endl;
    // range_results = rtree.range_search(2.0, 2.0, 5.0, 5.0);
    // for (const auto& point : range_results) {
    //     std::cout << "Found point: (" << point.x << ", " << point.y
    //               << ") ID: " << point.id << std::endl;
    // }

    // // Demonstrate bulk insertion
    // std::cout << "\nInserting additional points for stress testing..." << std::endl;
    // for (int i = 11; i <= 20; ++i) {
    //     double x = (i % 5) * 1.5;
    //     double y = (i / 5) * 1.5;
    //     rtree.insert(x, y, i);
    // }

    // std::cout << "Tree after bulk insertion:" << std::endl;
    // rtree.print_tree();

    return 0;
}