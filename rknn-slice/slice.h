#include "RStarTree2D.h"
#include <cmath>
#include <queue>
#include <set>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Global constants for partition calculations
// For 12 partitions, each covers 30 degrees (π/6 radians)
const double tan30 = 0.57735026919;  // tan(π/6) = 1/√3
const double tan60 = 1.73205080757;  // tan(π/3) = √3

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

// Helper function to normalize angle to [0, 2π)
inline double normalize_angle(double angle) {
    while (angle < 0) angle += 2 * M_PI;
    while (angle >= 2 * M_PI) angle -= 2 * M_PI;
    return angle;
}

// Helper function to calculate angular distance (shortest angular distance)
inline double angular_distance(double angle1, double angle2) {
    double diff = fabs(angle1 - angle2);
    if (diff > M_PI) {
        diff = 2 * M_PI - diff;
    }
    return diff;
}

// Structure to hold facility info with its bounds for sorting
struct FacilityBound {
    Point facility;
    double lower_bound;  // dist(f,q)/(2*cos(minAngle(f,P)))
    double upper_bound;  // dist(f,q)/(2*cos(maxAngle(f,P))) or INFINITY

    FacilityBound(const Point& f, double lower, double upper)
        : facility(f), lower_bound(lower), upper_bound(upper) {}

    // Sort by lower bound
    bool operator<(const FacilityBound& other) const {
        return lower_bound < other.lower_bound;
    }
};

// Angular partition for SLICE algorithm
struct AngularPartition {
    Point center;           // Center point (query point)
    double angle_start;     // Starting angle in radians [0, 2π)
    double angle_end;       // Ending angle in radians [0, 2π)
    int partition_id;       // Partition identifier (0 to 11 for 12 partitions)

    // SLICE algorithm specific
    double boundary_arc;    // k-th smallest upper bound (radius of the arc)
    vector<FacilityBound> sigList;  // Significant facilities sorted by lower bound
    priority_queue<double> upper_bounds_heap;  // Max-heap to maintain k smallest upper bounds

    AngularPartition(const Point& c, double start, double end, int id)
        : center(c), angle_start(start), angle_end(end), partition_id(id), boundary_arc(INFINITY) {}

    // Check if a point falls within this angular partition
    bool contains_point(const Point& p) const {
        double dx = p.x - center.x;
        double dy = p.y - center.y;

        // Special case: point is at center
        if (dx == 0 && dy == 0) {
            return true;  // Center belongs to all partitions conceptually
        }

        // Use the same strategy as getPartitionIndex
        // For 12 partitions, each covers 30 degrees (π/6 radians)
        // Using global tan30 and tan60 constants

        // Determine which partition the point belongs to using slopes
        int point_partition = 0;

        if (dx > 0) {  // Right half (partitions 0-2, 10-11)
            if (dy >= 0) {  // First quadrant (partitions 0-2)
                double slope = dy / dx;
                if (slope < tan30) {
                    point_partition = 0;
                } else if (slope < tan60) {
                    point_partition = 1;
                } else {
                    point_partition = 2;
                }
            } else {  // Fourth quadrant (partitions 10-11)
                double slope = -dy / dx;  // Make positive for comparison
                if (slope < tan30) {
                    point_partition = 11;
                } else if (slope < tan60) {
                    point_partition = 10;
                } else {
                    point_partition = 9;
                }
            }
        } else if (dx < 0) {  // Left half (partitions 3-8)
            if (dy >= 0) {  // Second quadrant (partitions 3-5)
                double slope = -dy / dx;  // Make positive for comparison
                if (slope < tan30) {
                    point_partition = 5;
                } else if (slope < tan60) {
                    point_partition = 4;
                } else {
                    point_partition = 3;
                }
            } else {  // Third quadrant (partitions 6-8)
                double slope = dy / dx;  // Both negative, so positive
                if (slope < tan30) {
                    point_partition = 6;
                } else if (slope < tan60) {
                    point_partition = 7;
                } else {
                    point_partition = 8;
                }
            }
        } else {  // dx == 0, point is directly above or below
            if (dy > 0) {
                point_partition = 2;  // Directly above (90°)
            } else {
                point_partition = 8;  // Directly below (270°)
            }
        }

        // Check if the calculated partition matches this partition
        return point_partition == partition_id;
    }

    // Get description of partition for debugging
    string get_description() const {
        double start_deg = angle_start * 180.0 / M_PI;
        double end_deg = angle_end * 180.0 / M_PI;
        return "Partition " + to_string(partition_id) +
               ": [" + to_string(start_deg) + "°, " + to_string(end_deg) + "°)";
    }
};

// Calculate the maximum subtended angle between a point x and a partition P
// The subtended angle is measured at the query point q
// Returns the maximum angle in radians
double maxAngle(const Point& query_point, const Point& x, const AngularPartition& partition) {
    // Calculate the angle from query_point to x
    double dx = x.x - query_point.x;
    double dy = x.y - query_point.y;
    double angle_to_x = normalize_angle(atan2(dy, dx));

    // The maximum subtended angle occurs between x and the partition boundary
    // that is furthest from x (angularly)

    // Calculate angular distances to both boundaries of the partition
    double dist_to_start = angular_distance(angle_to_x, partition.angle_start);
    double dist_to_end = angular_distance(angle_to_x, partition.angle_end);

    // The maximum subtended angle is the larger of the two angular distances
    double max_subtended_angle = max(dist_to_start, dist_to_end);

    // Special case: if the partition wraps around (crosses 0/2π boundary)
    if (partition.angle_start > partition.angle_end) {
        // For wrap-around partitions, we need to consider the angular span differently
        double partition_span = (2 * M_PI - partition.angle_start) + partition.angle_end;

        // If x is inside the partition
        if (partition.contains_point(x)) {
            // Maximum angle is to the furthest boundary
            max_subtended_angle = max(dist_to_start, dist_to_end);
        } else {
            // If x is outside, we need to find the maximum angle to any point in the partition
            // This occurs at one of the boundaries
            max_subtended_angle = max(dist_to_start, dist_to_end);
        }
    }

    return max_subtended_angle;
}

// Alternative implementation that considers the entire angular span of the partition
double maxAngleSpan(const Point& query_point, const Point& x, const AngularPartition& partition) {
    // Calculate the angle from query_point to x
    double dx = x.x - query_point.x;
    double dy = x.y - query_point.y;
    double angle_to_x = normalize_angle(atan2(dy, dx));

    // Calculate the angular span of the partition
    double partition_span;
    if (partition.angle_end >= partition.angle_start) {
        partition_span = partition.angle_end - partition.angle_start;
    } else {
        // Wrap-around case
        partition_span = (2 * M_PI - partition.angle_start) + partition.angle_end;
    }

    // If point x is within the partition
    if (partition.contains_point(x)) {
        // Maximum subtended angle is from x to the furthest boundary
        double dist_to_start = angular_distance(angle_to_x, partition.angle_start);
        double dist_to_end = angular_distance(angle_to_x, partition.angle_end);
        return max(dist_to_start, dist_to_end);
    } else {
        // If point x is outside the partition
        // Maximum subtended angle is from x to the furthest point in the partition

        // Calculate distances to both boundaries
        double dist_to_start = angular_distance(angle_to_x, partition.angle_start);
        double dist_to_end = angular_distance(angle_to_x, partition.angle_end);

        // The maximum occurs at the boundary that is angularly furthest from x
        return max(dist_to_start, dist_to_end);
    }
}

// Calculate the minimum subtended angle between a point x and a partition P
// The subtended angle is measured at the query point q
// Returns the minimum angle in radians
double minAngle(const Point& query_point, const Point& x, const AngularPartition& partition) {
    // Calculate the angle from query_point to x
    double dx = x.x - query_point.x;
    double dy = x.y - query_point.y;
    double angle_to_x = normalize_angle(atan2(dy, dx));

    // If point x is within the partition, minimum angle is 0
    if (partition.contains_point(x)) {
        return 0.0;
    }

    // If x is outside the partition, the minimum subtended angle
    // occurs between x and the closest boundary of the partition

    // Calculate angular distances to both boundaries
    double dist_to_start = angular_distance(angle_to_x, partition.angle_start);
    double dist_to_end = angular_distance(angle_to_x, partition.angle_end);

    // The minimum subtended angle is the smaller of the two angular distances
    double min_subtended_angle = min(dist_to_start, dist_to_end);

    // Special case: check if x is between the boundaries (for wrap-around partitions)
    if (partition.angle_start > partition.angle_end) {
        // Wrap-around partition case
        // x is "between" boundaries if it's NOT in the partition
        // but would be if we inverted the partition
        bool x_in_gap = (angle_to_x > partition.angle_end && angle_to_x < partition.angle_start);

        if (x_in_gap) {
            // x is in the gap between end and start
            // minimum distance is to the closest boundary
            min_subtended_angle = min(dist_to_start, dist_to_end);
        }
    }

    return min_subtended_angle;
}

// Alternative implementation with more explicit logic
double minAngleAlt(const Point& query_point, const Point& x, const AngularPartition& partition) {
    // If x is at the same position as query_point, angle is undefined
    double dx = x.x - query_point.x;
    double dy = x.y - query_point.y;

    if (dx == 0 && dy == 0) {
        return 0.0; // or could return NaN or handle specially
    }

    double angle_to_x = normalize_angle(atan2(dy, dx));

    // If point x is within the partition, the minimum angle is 0
    // because x itself is a point in the partition
    if (partition.contains_point(x)) {
        return 0.0;
    }

    // For x outside the partition, find minimum angular distance to partition
    double dist_to_start = angular_distance(angle_to_x, partition.angle_start);
    double dist_to_end = angular_distance(angle_to_x, partition.angle_end);

    // The minimum angle is to the nearer boundary
    return min(dist_to_start, dist_to_end);
}

// Helper function to check if an angle is between two angles (considering wrap-around)
bool isAngleBetween(double angle, double start, double end) {
    angle = normalize_angle(angle);
    start = normalize_angle(start);
    end = normalize_angle(end);

    if (start <= end) {
        return angle >= start && angle <= end;
    } else {
        // Wrap-around case
        return angle >= start || angle <= end;
    }
}

// Create 12 equally sized angular partitions around a query point
vector<AngularPartition> create_angular_partitions(const Point& query_point, int num_partitions = 12) {
    vector<AngularPartition> partitions;
    double angle_per_partition = (2 * M_PI) / num_partitions;

    for (int i = 0; i < num_partitions; i++) {
        double start_angle = i * angle_per_partition;
        double end_angle = ((i + 1) % num_partitions) * angle_per_partition;

        // Handle the last partition that wraps around
        if (i == num_partitions - 1) {
            end_angle = 2 * M_PI;
        }

        partitions.push_back(AngularPartition(query_point, start_angle, end_angle, i));
    }

    return partitions;
}

// Helper function to calculate minimum distance from point to rectangle
inline double min_distance_to_rect(const Point& p, const Rectangle& rect) {
    double dx = 0, dy = 0;
    if (p.x < rect.min_x) dx = rect.min_x - p.x;
    else if (p.x > rect.max_x) dx = p.x - rect.max_x;
    if (p.y < rect.min_y) dy = rect.min_y - p.y;
    else if (p.y > rect.max_y) dy = p.y - rect.max_y;
    return sqrt(dx * dx + dy * dy);
}

// Calculate the points M and N where boundary arc intersects partition boundaries
pair<Point, Point> getBoundaryArcIntersections(const AngularPartition& partition, double arc_radius) {
    // M is at angle_start, distance arc_radius from center
    Point M(partition.center.x + arc_radius * cos(partition.angle_start),
            partition.center.y + arc_radius * sin(partition.angle_start));

    // N is at angle_end, distance arc_radius from center
    Point N(partition.center.x + arc_radius * cos(partition.angle_end),
            partition.center.y + arc_radius * sin(partition.angle_end));

    return make_pair(M, N);
}

// Check if facility f is significant for partition P using Lemma 1 and Lemma 2
bool isSignificantFacility(const Point& f, const Point& q, const AngularPartition& partition) {
    double dist_fq = f.distance_to(q);

    // If boundary arc is not set (INFINITY), we cannot prune, so keep the facility
    if (partition.boundary_arc == INFINITY) {
        return true;  // Keep as potentially significant when no boundary is set
    }

    // Lemma 1: if f is in P and dist(f,q) > 2 * boundary_arc, f is not significant
    if (partition.contains_point(f)) {
        if (dist_fq > 2.0 * partition.boundary_arc) {
            return false;
        }
        return true;  // f is in P and close enough
    }

    // Lemma 2: if f is not in P, check distances to M and N
    pair<Point, Point> arc_intersections = getBoundaryArcIntersections(partition, partition.boundary_arc);
    Point M = arc_intersections.first;
    Point N = arc_intersections.second;
    double dist_Mf = M.distance_to(f);
    double dist_Nf = N.distance_to(f);

    // f is not significant if both distances are greater than boundary_arc
    if (dist_Mf > partition.boundary_arc && dist_Nf > partition.boundary_arc) {
        return false;
    }

    return true;  // f is significant
}

// Check if a node/MBR may contain a significant facility for at least one partition
bool mayContainSignificantFacility(const Rectangle& mbr,
                                   const vector<AngularPartition>& partitions,
                                   const Point& query_point) {
    // Check each partition
    for (const auto& partition : partitions) {
        // If boundary arc is not set, we can't prune
        if (partition.boundary_arc == INFINITY) {
            return true;
        }

        // Get the four corners of the MBR
        Point corners[4] = {
            Point(mbr.min_x, mbr.min_y),
            Point(mbr.max_x, mbr.min_y),
            Point(mbr.min_x, mbr.max_y),
            Point(mbr.max_x, mbr.max_y)
        };

        // Check if any corner could be significant for this partition
        for (const auto& corner : corners) {
            double dist_to_q = corner.distance_to(query_point);

            // Check Lemma 1: if corner is in partition
            if (partition.contains_point(corner)) {
                // Corner could be significant if dist <= 2 * boundary_arc
                if (dist_to_q <= 2.0 * partition.boundary_arc) {
                    return true;  // MBR may contain significant facility
                }
            } else {
                // Check Lemma 2: if corner is not in partition
                pair<Point, Point> arc_intersections = getBoundaryArcIntersections(partition, partition.boundary_arc);
                Point M = arc_intersections.first;
                Point N = arc_intersections.second;
                double dist_Mc = M.distance_to(corner);
                double dist_Nc = N.distance_to(corner);

                // If either distance is <= boundary_arc, MBR might contain significant facility
                if (dist_Mc <= partition.boundary_arc || dist_Nc <= partition.boundary_arc) {
                    return true;
                }
            }
        }

        // Also check if the MBR might intersect with the partition's significant region
        // Conservative check: if MBR is close enough to query point
        double min_dist_mbr_to_q = min_distance_to_rect(query_point, mbr);
        if (min_dist_mbr_to_q <= 2.0 * partition.boundary_arc) {
            return true;
        }
    }

    return false;  // MBR cannot contain significant facility for any partition
}

// Main pruneSpace function to process a single facility and update partitions
void pruneSpace(vector<AngularPartition>& partitions,
                const Point& f,  // Single facility to process
                const Point& query_point,
                int k) {

    const double NINETY_DEGREES = M_PI / 2.0;

    // Skip if facility is the query point itself
    if (f.id == query_point.id) {
        return;
    }

    // Process each partition for this facility
    for (auto& partition : partitions) {
        double min_ang = minAngle(query_point, f, partition);

        // Only consider this facility if minAngle < 90 degrees
        if (min_ang < NINETY_DEGREES) {
            double max_ang = maxAngle(query_point, f, partition);
            double dist_fq = f.distance_to(query_point);

            // Calculate upper bound for this facility
            double upper_bound;
            if (max_ang >= NINETY_DEGREES) {
                // Upper bound is infinite
                upper_bound = INFINITY;
            } else {
                // Upper bound = dist(f,q) / (2*cos(maxAngle))
                upper_bound = dist_fq / (2.0 * cos(max_ang));
            }

            // Use max-heap to maintain k smallest upper bounds
            if (partition.upper_bounds_heap.size() < k) {
                // If we have fewer than k elements, just add to heap
                partition.upper_bounds_heap.push(upper_bound);
                // Update boundary_arc to the largest in heap (k-th smallest so far)
                partition.boundary_arc = partition.upper_bounds_heap.top();
            } else {
                // If we have k elements, only add if new bound is smaller than max
                if (upper_bound < partition.upper_bounds_heap.top()) {
                    partition.upper_bounds_heap.pop();  // Remove largest
                    partition.upper_bounds_heap.push(upper_bound);  // Add new smaller value
                    partition.boundary_arc = partition.upper_bounds_heap.top();  // Update to new k-th smallest
                }
                // If upper_bound >= top, it doesn't affect k-th smallest, so don't add it
            }

            // Check if f is significant using lemmas
            if (isSignificantFacility(f, query_point, partition)) {
                // Calculate lower bound
                double lower_bound;
                if (min_ang == 0) {
                    // f is in the partition, lower bound is just half distance
                    lower_bound = dist_fq / 2.0;
                } else {
                    lower_bound = dist_fq / (2.0 * cos(min_ang));
                }

                // Add to sigList
                partition.sigList.push_back(FacilityBound(f, lower_bound, upper_bound));

                // Keep sigList sorted by lower bound
                sort(partition.sigList.begin(), partition.sigList.end());
            }
        }
    }
}

// Helper function to get partition index using relative positions
// Avoids expensive trigonometric calculations
inline int getPartitionIndex(double dx, double dy) {
    // Handle special cases
    if (dx == 0 && dy == 0) return 0;  // Point at query location

    // For 12 partitions, each covers 30 degrees (π/6 radians)
    // We can determine partition using slopes and quadrants

    // Determine quadrant and calculate partition
    // Partition 0: angle [0, π/6), Partition 1: [π/6, π/3), etc.

    int partition_idx = 0;

    // Use slopes to determine partition without computing actual angle
    // Using global tan30 and tan60 constants

    if (dx > 0) {  // Right half (partitions 0-2, 10-11)
        if (dy >= 0) {  // First quadrant (partitions 0-2)
            double slope = dy / dx;
            if (slope < tan30) {
                partition_idx = 0;
            } else if (slope < tan60) {
                partition_idx = 1;
            } else {
                partition_idx = 2;
            }
        } else {  // Fourth quadrant (partitions 10-11)
            double slope = -dy / dx;  // Make positive for comparison
            if (slope < tan30) {
                partition_idx = 11;
            } else if (slope < tan60) {
                partition_idx = 10;
            } else {
                partition_idx = 9;
            }
        }
    } else if (dx < 0) {  // Left half (partitions 3-8)
        if (dy >= 0) {  // Second quadrant (partitions 3-5)
            double slope = -dy / dx;  // Make positive for comparison
            if (slope < tan30) {
                partition_idx = 5;
            } else if (slope < tan60) {
                partition_idx = 4;
            } else {
                partition_idx = 3;
            }
        } else {  // Third quadrant (partitions 6-8)
            double slope = dy / dx;  // Both negative, so positive
            if (slope < tan30) {
                partition_idx = 6;
            } else if (slope < tan60) {
                partition_idx = 7;
            } else {
                partition_idx = 8;
            }
        }
    } else {  // dx == 0, point is directly above or below
        if (dy > 0) {
            partition_idx = 3;  // Directly above (90 degrees)
        } else {
            partition_idx = 9;  // Directly below (270 degrees)
        }
    }

    return partition_idx;
}

// Check if a user u is a reverse k-nearest neighbor of query q
bool isRkNN(const Point& u, double dist_uq,
            const AngularPartition& user_partition, int k) {
    // This function now takes only the specific partition that u lies in
    // and the pre-calculated distance from u to q
    int count = 0;

    // Check facilities in sigList in ascending order of lower bound
    for (const auto& fb : user_partition.sigList) {
        // If dist(u,q) <= lower bound of f to P, u is RkNN
        if (dist_uq <= fb.lower_bound) {
            return true;
        }

        // Check if this facility is closer to u than q is
        double dist_uf = u.distance_to(fb.facility);
        if (dist_uf < dist_uq) {
            count++;
            if (count >= k) {
                return false;  // u has k closer facilities than q
            }
        }
    }

    return true;  // u is RkNN
}

// Check if a rectangle (MBR) lies completely in the pruned area
bool isCompletelyPruned(const Rectangle& mbr,
                        const vector<AngularPartition>& partitions,
                        const Point& query_point) {
    // If query point is inside the MBR, it cannot be pruned
    if (query_point.x >= mbr.min_x && query_point.x <= mbr.max_x &&
        query_point.y >= mbr.min_y && query_point.y <= mbr.max_y) {
        return false;
    }

    // Get the four corners of the MBR
    Point corners[4] = {
        Point(mbr.min_x, mbr.min_y),
        Point(mbr.max_x, mbr.min_y),
        Point(mbr.min_x, mbr.max_y),
        Point(mbr.max_x, mbr.max_y)
    };

    // Find which partitions the MBR overlaps with
    // An MBR overlaps with a partition if any of its corners are in that partition
    // or if the MBR crosses through the partition

    // First, find which partitions contain each corner
    set<int> overlapping_partitions;
    for (int i = 0; i < 4; i++) {
        double dx = corners[i].x - query_point.x;
        double dy = corners[i].y - query_point.y;
        int partition_idx = getPartitionIndex(dx, dy);
        overlapping_partitions.insert(partition_idx);
    }

    // Also check if MBR spans across the query point
    // If it does, it might overlap with additional partitions
    bool spans_horizontally = (mbr.min_x < query_point.x) && (mbr.max_x > query_point.x);
    bool spans_vertically = (mbr.min_y < query_point.y) && (mbr.max_y > query_point.y);

    if (spans_horizontally || spans_vertically) {
        // MBR crosses through the query point region
        // Need to check edges for additional partition overlaps

        // Check top and bottom edges if spans horizontally
        if (spans_horizontally) {
            // Top edge at query.x
            double dx = 0;
            double dy_top = mbr.max_y - query_point.y;
            double dy_bottom = mbr.min_y - query_point.y;

            if (dy_top > 0) {
                overlapping_partitions.insert(getPartitionIndex(0, dy_top));
            }
            if (dy_bottom < 0) {
                overlapping_partitions.insert(getPartitionIndex(0, dy_bottom));
            }
        }

        // Check left and right edges if spans vertically
        if (spans_vertically) {
            // Side edges at query.y
            double dy = 0;
            double dx_right = mbr.max_x - query_point.x;
            double dx_left = mbr.min_x - query_point.x;

            if (dx_right > 0) {
                overlapping_partitions.insert(getPartitionIndex(dx_right, 0));
            }
            if (dx_left < 0) {
                overlapping_partitions.insert(getPartitionIndex(dx_left, 0));
            }
        }
    }

    // Now check if MBR is pruned for all overlapping partitions
    double min_dist_to_q = min_distance_to_rect(query_point, mbr);

    for (int partition_idx : overlapping_partitions) {
        const AngularPartition& partition = partitions[partition_idx];

        // If this partition has infinite boundary arc, cannot prune
        if (partition.boundary_arc == INFINITY) {
            return false;
        }

        // If MBR's closest point is within this partition's boundary arc, cannot prune
        if (min_dist_to_q <= partition.boundary_arc) {
            return false;
        }
    }

    // MBR is beyond the boundary arc of all partitions it overlaps with
    return true;
}

// Traverse user R*-tree to find RkNN candidates
void traverseUserTree(shared_ptr<RStarNode> node,
                     const vector<AngularPartition>& partitions,
                     const Point& query_point,
                     vector<Point>& rknn_results,
                     int k) {
    if (!node) return;

    // Check if this node's MBR is completely in pruned area
    Rectangle mbr = node->get_mbr();
    if (isCompletelyPruned(mbr, partitions, query_point)) {
        return;  // Skip this entire subtree
    }

    if (node->is_leaf) {
        // Process each user point in this leaf
        for (const auto& entry : node->entries) {
            // Check if this user point is in pruned area
            bool is_pruned = false;

            // Directly calculate which partition this user belongs to using relative positions
            double dx = entry.point.x - query_point.x;
            double dy = entry.point.y - query_point.y;

            int partition_idx = getPartitionIndex(dx, dy);

            const auto& partition = partitions[partition_idx];

            // Check if user is in pruned area of this partition
            double dist_to_q = entry.point.distance_to(query_point);
            if (dist_to_q > partition.boundary_arc) {
                is_pruned = true;
                // cout << "prune a point" << endl;
            }

            if (!is_pruned) {
                // Verify if this user is RkNN
                // Pass only the partition that the user lies in and the pre-calculated distance
                if (isRkNN(entry.point, dist_to_q, partition, k)) {
                    rknn_results.push_back(entry.point);
                }
            }
        }
    } else {
        // Internal node: recursively traverse children
        for (const auto& entry : node->entries) {
            if (entry.child) {
                traverseUserTree(entry.child, partitions, query_point, rknn_results, k);
            }
        }
    }
}

// void dfs_rknn_traverse(shared_ptr<RStarNode> node, const vector<Line>& bisectors,
//                        vector<Point>& rknn_candidates, int k) {
//     if (!node) return;

//     if (node->is_leaf) {
//         // For leaf nodes, check each point
//         for (const auto& entry : node->entries) {
//             int violations = 0;

//             // Count how many bisectors this point violates (not on valid side)
//             for (const auto& bisector : bisectors) {
//                 if (!bisector.is_on_valid_side(entry.point)) {
//                     violations++;
//                     if (violations >= k) {
//                         break; // Early termination if violations exceed k
//                     }
//                 }
//             }

//             // If violations <= k, this point is a candidate
//             if (violations < k) {
//                 rknn_candidates.push_back(entry.point);
//             }
//         }
//     } else {
//         // For internal nodes, check if we can prune the subtree
//         for (const auto& entry : node->entries) {
//             int violations = 0;

//             // Count how many bisectors this node's MBR violates
//             for (const auto& bisector : bisectors) {
//                 if (can_prune_node_by_valid_side(entry.child, bisector)) {
//                     violations++;
//                     if (violations >= k) {
//                         break; // Early termination if violations exceed k
//                     }
//                 }
//             }

//             // If violations < k, continue DFS on this child
//             if (violations < k) {
//                 dfs_rknn_traverse(entry.child, bisectors, rknn_candidates, k);
//             }
//         }
//     }
// }

// void get_rknn_candidates(
//     RStarTree* rtree,
//     const vector<Line>& bisectors,
//     vector<Point>& rknn_candidates,
//     int k
// ) {
//     rknn_candidates.clear();
//     dfs_rknn_traverse(rtree->get_root(), bisectors, rknn_candidates, k);
// }

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
