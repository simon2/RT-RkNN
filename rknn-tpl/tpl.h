#include "RStarTree2D.h"
#include <cmath>

// 2D Line class
class Line {
public:
    double a, b;     // For y = ax + b (non-vertical lines)
    double x_val;    // For vertical lines: x = x_val
    bool is_vertical;
    int valid_side;  // 1 for above(when vertical: right), 0 for below(when vertical: left)

    Line() : a(0), b(0), x_val(0), is_vertical(false), valid_side(1) {}

    // Constructor from two points
    Line(const Point& p1, const Point& p2) {
        if (p1.x == p2.x) {
            // Vertical line: x = constant
            is_vertical = true;
            x_val = p1.x;
            a = 0;
            b = 0;
        } else {
            // Non-vertical line: y = ax + b
            is_vertical = false;
            a = (p2.y - p1.y) / (p2.x - p1.x);
            b = p1.y - a * p1.x;
            x_val = 0;
        }
        valid_side = 1;  // Default to above/right as valid side
    }

    // Constructor from slope-intercept form: y = mx + b
    Line(double slope, double y_intercept) {
        if (std::isinf(slope)) {
            is_vertical = true;
            x_val = y_intercept;  // In this case, y_intercept represents x-coordinate
            a = 0;
            b = 0;
        } else {
            is_vertical = false;
            a = slope;
            b = y_intercept;
            x_val = 0;
        }
        valid_side = 1;  // Default to above/right as valid side
    }

    // Constructor from coefficients (convert ax + by + c = 0 to our format)
    Line(double coeff_a, double coeff_b, double coeff_c) {
        if (coeff_b == 0) {
            // Vertical line: ax + c = 0 -> x = -c/a
            is_vertical = true;
            x_val = -coeff_c / coeff_a;
            a = 0;
            b = 0;
        } else {
            // Non-vertical line: ax + by + c = 0 -> y = (-a/b)x + (-c/b)
            is_vertical = false;
            a = -coeff_a / coeff_b;
            b = -coeff_c / coeff_b;
            x_val = 0;
        }
        valid_side = 1;  // Default to above/right as valid side
    }

    // Evaluate line equation at point
    double evaluate(const Point& point) const {
        if (is_vertical) {
            return point.x - x_val;  // Distance from vertical line
        } else {
            return point.y - (a * point.x + b);  // Distance from y = ax + b
        }
    }

    // Check if point is above the line (positive side)
    bool is_above(const Point& point) const {
        return evaluate(point) > 0;
    }

    // Check if point is below the line (negative side)
    bool is_below(const Point& point) const {
        return evaluate(point) < 0;
    }

    // Check if point is on the line
    bool is_on_line(const Point& point, double epsilon = 1e-9) const {
        return abs(evaluate(point)) < epsilon;
    }

    // Distance from point to line
    double distance_to_point(const Point& point) const {
        if (is_vertical) {
            return abs(point.x - x_val);
        } else {
            return abs(point.y - (a * point.x + b)) / sqrt(a * a + 1);
        }
    }

    // Set the valid side (1 for above/right, 0 for below/left)
    void set_valid_side(int side) {
        valid_side = side;
    }

    // Check if point is on the valid side for query processing
    bool is_on_valid_side(const Point& point) const {
        if (valid_side == 1) {
            // Valid side is above (for non-vertical) or right (for vertical)
            if (is_vertical) {
                return point.x >= x_val;  // Right side of vertical line
            } else {
                return point.y >= (a * point.x + b);  // Above or on the line
            }
        } else {
            // Valid side is below (for non-vertical) or left (for vertical)
            if (is_vertical) {
                return point.x <= x_val;  // Left side of vertical line
            } else {
                return point.y <= (a * point.x + b);  // Below or on the line
            }
        }
    }
};

// Line position enumeration
enum class LinePosition {
    ALL_ABOVE,    // All points of rectangle are above the line
    ALL_BELOW,    // All points of rectangle are below the line
    INTERSECTS    // Rectangle intersects or straddles the line
};

// Check if a rectangle is entirely on one side of a line
LinePosition check_rectangle_line_position(const Rectangle& rect, const Line& line) {
    // Check all four corners of the rectangle
    Point corners[4] = {
        Point(rect.min_x, rect.min_y),  // bottom-left
        Point(rect.max_x, rect.min_y),  // bottom-right
        Point(rect.min_x, rect.max_y),  // top-left
        Point(rect.max_x, rect.max_y)   // top-right
    };

    int above_count = 0;
    int below_count = 0;

    for (int i = 0; i < 4; i++) {
        double eval = line.evaluate(corners[i]);
        if (eval > 0) {
            above_count++;
        } else if (eval < 0) {
            below_count++;
        }
        // Points exactly on the line are considered as intersecting
    }

    if (above_count == 4) {
        return LinePosition::ALL_ABOVE;
    } else if (below_count == 4) {
        return LinePosition::ALL_BELOW;
    } else {
        return LinePosition::INTERSECTS;
    }
}

// Convenience function to check if RStarTree node is all above the line
bool is_node_all_above_line(shared_ptr<RStarNode> node, const Line& line) {
    Rectangle mbr = node->get_mbr();
    return check_rectangle_line_position(mbr, line) == LinePosition::ALL_ABOVE;
}

// Convenience function to check if RStarTree node is all below the line
bool is_node_all_below_line(shared_ptr<RStarNode> node, const Line& line) {
    Rectangle mbr = node->get_mbr();
    return check_rectangle_line_position(mbr, line) == LinePosition::ALL_BELOW;
}

// Function to check if node can be pruned based on line constraint
bool can_prune_node_by_line(shared_ptr<RStarNode> node, const Line& line, bool want_above) {
    Rectangle mbr = node->get_mbr();
    LinePosition pos = check_rectangle_line_position(mbr, line);

    if (want_above) {
        return pos == LinePosition::ALL_BELOW;  // Prune if all below when we want above
    } else {
        return pos == LinePosition::ALL_ABOVE;  // Prune if all above when we want below
    }
}

// Function to check if node can be pruned based on line's valid side
bool can_prune_node_by_valid_side(shared_ptr<RStarNode> node, const Line& line) {
    Rectangle mbr = node->get_mbr();
    LinePosition pos = check_rectangle_line_position(mbr, line);

    if (line.valid_side == 1) {
        // Valid side is above/right, prune if all points are below/left
        return pos == LinePosition::ALL_BELOW;
    } else {
        // Valid side is below/left, prune if all points are above/right
        return pos == LinePosition::ALL_ABOVE;
    }
}

// Function to calculate the perpendicular bisector of two points
Line perpendicular_bisector(const Point& p1, const Point& p2) { // p1 should be the query point
    // Midpoint of the two points
    double mid_x = (p1.x + p2.x) / 2.0;
    double mid_y = (p1.y + p2.y) / 2.0;

    Line bisector;

    // Handle vertical line case (p1.x == p2.x)
    if (p1.x == p2.x) {
        // Original line is vertical, perpendicular bisector is horizontal
        // Horizontal line: y = mid_y (slope = 0, y-intercept = mid_y)
        bisector = Line(0, mid_y);
    }
    // Handle horizontal line case (p1.y == p2.y)
    else if (p1.y == p2.y) {
        // Original line is horizontal, perpendicular bisector is vertical
        // Vertical line: x = mid_x
        bisector = Line(INFINITY, mid_x);
    }
    else {
        // Calculate slope of original line
        double original_slope = (p2.y - p1.y) / (p2.x - p1.x);

        // Perpendicular slope is negative reciprocal
        double perp_slope = -1.0 / original_slope;

        // Calculate y-intercept: y = mx + b -> b = y - mx
        double y_intercept = mid_y - perp_slope * mid_x;

        bisector = Line(perp_slope, y_intercept);
    }

    // Set valid side based on where p1 is relative to the bisector
    if (bisector.is_vertical) {
        // For vertical line, check if p1 is to the right (1) or left (0)
        bisector.valid_side = (p1.x >= bisector.x_val) ? 1 : 0;
    } else {
        // For non-vertical line, check if p1 is above (1) or below (0)
        bisector.valid_side = (p1.y >= (bisector.a * p1.x + bisector.b)) ? 1 : 0;
    }

    return bisector;
}

void dfs_rknn_traverse(shared_ptr<RStarNode> node, const vector<Line>& bisectors,
                       vector<Point>& rknn_candidates, int k) {
    if (!node) return;

    if (node->is_leaf) {
        // For leaf nodes, check each point
        for (const auto& entry : node->entries) {
            int violations = 0;

            // Count how many bisectors this point violates (not on valid side)
            for (const auto& bisector : bisectors) {
                if (!bisector.is_on_valid_side(entry.point)) {
                    violations++;
                    if (violations > k) {
                        break; // Early termination if violations exceed k
                    }
                }
            }

            // If violations <= k, this point is a candidate
            if (violations <= k) {
                rknn_candidates.push_back(entry.point);
            }
        }
    } else {
        // For internal nodes, check if we can prune the subtree
        for (const auto& entry : node->entries) {
            int violations = 0;

            // Count how many bisectors this node's MBR violates
            for (const auto& bisector : bisectors) {
                if (can_prune_node_by_valid_side(entry.child, bisector)) {
                    violations++;
                    if (violations > k) {
                        break; // Early termination if violations exceed k
                    }
                }
            }

            // If violations <= k, continue DFS on this child
            if (violations <= k) {
                dfs_rknn_traverse(entry.child, bisectors, rknn_candidates, k);
            }
        }
    }
}

void get_rknn_candidates(
    RStarTree* rtree,
    const vector<Line>& bisectors,
    vector<Point>& rknn_candidates,
    int k
) {
    rknn_candidates.clear();
    dfs_rknn_traverse(rtree->get_root(), bisectors, rknn_candidates, k);
}

// Priority Queue Entry for R*-tree traversal with bisector creation
struct PQEntry {
    shared_ptr<RStarNode> node;
    double distance;  // Distance from query point to node's MBR or point
    bool is_point;
    Point point;  // Used when is_point is true

    // Priority queue uses max-heap by default, so we invert comparison for min-heap
    bool operator>(const PQEntry& other) const {
        return distance > other.distance;
    }
};

// Helper function to calculate minimum distance from point to rectangle
inline double min_distance_to_rect(const Point& p, const Rectangle& rect) {
    double dx = 0, dy = 0;
    if (p.x < rect.min_x) dx = rect.min_x - p.x;
    else if (p.x > rect.max_x) dx = p.x - rect.max_x;
    if (p.y < rect.min_y) dy = rect.min_y - p.y;
    else if (p.y > rect.max_y) dy = p.y - rect.max_y;
    return sqrt(dx * dx + dy * dy);
}