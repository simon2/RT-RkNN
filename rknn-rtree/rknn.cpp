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

#include "rknn.h"

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

        cout << "Naive RkNN search with R*-tree..." << endl;
        start_time = get_wall_time();
        Point query_point(fac[q].x, fac[q].y, fac[q].id);
        vector<Point> naive_rknn;
        for (uint32_t i = 0; i < usr_cnt; ++i) {
            const auto& candidate = usr[i];
            auto fac_knn_of_candidate = fac_rtree.knn_search(candidate, k);
            bool found = false;
            for (const auto& neighbor : fac_knn_of_candidate) {
                if (query_point.id == neighbor.id) {
                    found = true;
                    break;
                }
            }
            if (found) {
                naive_rknn.push_back(candidate);
            }
        }
        end_time = get_wall_time();
        cout << "Found " << naive_rknn.size() << " RkNN results." << endl;
        cout << "Verification time: " << end_time - start_time << "[s]." << endl << endl;
    }
    catch( exception& e )
    {
        cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}