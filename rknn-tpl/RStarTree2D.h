#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <memory>
#include <queue>

using namespace std;

// Forward declarations
class Rectangle;

// 2D Point class
class Point {
public:
    double x, y;
    int id;  // Point identifier

    Point() : x(0), y(0), id(-1) {}
    Point(double x, double y, int id = -1) : x(x), y(y), id(id) {}

    double distance_to(const Point& other) const;
    bool operator<(const Point& other) const;
    Rectangle to_rectangle() const;
};

// Rectangle class for bounding boxes (MBRs)
class Rectangle {
public:
    double min_x, min_y, max_x, max_y;

    Rectangle() : min_x(0), min_y(0), max_x(0), max_y(0) {}
    Rectangle(double min_x, double min_y, double max_x, double max_y)
        : min_x(min_x), min_y(min_y), max_x(max_x), max_y(max_y) {}

    double area() const;
    double perimeter() const;
    bool intersects(const Rectangle& other) const;
    bool contains(const Rectangle& other) const;
    bool contains_point(const Point& point) const;
    Rectangle union_with(const Rectangle& other) const;
    Rectangle union_with_point(const Point& point) const;
    double enlargement_area(const Rectangle& other) const;
    double enlargement_area_point(const Point& point) const;
};

// Forward declaration
class RStarNode;

// Entry in a node (can be point data or child node)
class RStarEntry {
public:
    Rectangle mbr;  // Minimum Bounding Rectangle
    shared_ptr<RStarNode> child;  // Child node (null for leaf entries)
    Point point;  // Point data (for leaf entries)

    // Constructor for leaf entry (point)
    RStarEntry(const Point& pt);

    // Constructor for internal node entry
    RStarEntry(const Rectangle& rect, shared_ptr<RStarNode> node);

    bool is_leaf_entry() const;
};

// R* Tree Node
class RStarNode {
public:
    vector<RStarEntry> entries;
    bool is_leaf;
    shared_ptr<RStarNode> parent;

    static const int MIN_ENTRIES = 2;
    static const int MAX_ENTRIES = 8;

    RStarNode(bool leaf = false);

    Rectangle get_mbr() const;
    bool is_full() const;
    bool is_underflow() const;
};

// R* Tree implementation for 2D points
class RStarTree {
private:
    shared_ptr<RStarNode> root;
    int reinsert_count;
    static const int REINSERT_FACTOR = 30;  // 30% for reinsertion

public:
    RStarTree();

    void insert(const Point& point);
    void insert(double x, double y, int id);

    // Search for points within a rectangular region
    vector<Point> range_search(const Rectangle& query_rect);
    vector<Point> range_search(double min_x, double min_y, double max_x, double max_y);

    // Find nearest neighbors within a given radius
    vector<Point> radius_search(const Point& query_point, double radius);

    // Find k nearest neighbors
    vector<Point> knn_search(const Point& query_point, int k);

    bool remove(const Point& point);
    bool remove(double x, double y, int id);

    void print_tree();

private:
    void insert_entry(RStarEntry& entry, int level);

    shared_ptr<RStarNode> choose_subtree(shared_ptr<RStarNode> node,
                                             const RStarEntry& entry,
                                             int target_level,
                                             int current_level = 0);

    // Keep the old choose_leaf for backward compatibility (points only)
    shared_ptr<RStarNode> choose_leaf(shared_ptr<RStarNode> node,
                                          const Point& point, int target_level);

    double calculate_overlap_enlargement_point(shared_ptr<RStarNode> node,
                                             const RStarEntry& entry, const Point& point);

    double calculate_overlap_enlargement_rect(shared_ptr<RStarNode> node,
                                            const RStarEntry& entry, const Rectangle& rect);

    int get_node_level(shared_ptr<RStarNode> node);

    void reinsert(shared_ptr<RStarNode> node, RStarEntry& new_entry);

    shared_ptr<RStarNode> split_node(shared_ptr<RStarNode> node, RStarEntry& new_entry);

    int choose_split_axis(shared_ptr<RStarNode> node);
    double calculate_axis_goodness(shared_ptr<RStarNode> node, int axis);
    int choose_split_index(shared_ptr<RStarNode> node, int axis);
    void sort_entries(shared_ptr<RStarNode> node, int axis);
    Rectangle get_mbr_for_entries(shared_ptr<RStarNode> node, int start, int end);

    void insert_into_parent(shared_ptr<RStarNode> parent, RStarEntry& entry);
    void adjust_tree(shared_ptr<RStarNode> node);

    void range_search_recursive(shared_ptr<RStarNode> node, const Rectangle& query_rect,
                               vector<Point>& results);

    void knn_search_recursive(shared_ptr<RStarNode> node, const Point& query_point,
                             int k, priority_queue<pair<double, Point>>& pq);

    double min_distance_to_rect(const Point& point, const Rectangle& rect);

    bool remove_point(shared_ptr<RStarNode> node, const Point& point);
    void handle_underflow(shared_ptr<RStarNode> node);
    void print_node(shared_ptr<RStarNode> node, int depth);
};
