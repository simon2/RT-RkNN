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

// Forward declaration
inline int getPartitionIndex(double dx, double dy);

// Helper function to normalize angle to [0, 2π)
inline double normalize_angle(double angle) {
    angle = fmod(angle, 2 * M_PI);
    if (angle < 0) angle += 2 * M_PI;
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

    // Cached cos/sin values for boundary arc intersection calculations
    double cos_start, sin_start;
    double cos_end, sin_end;
    // Cached M and N points (boundary arc intersections)
    Point cached_M, cached_N;
    double cached_arc_radius;  // The radius used to compute cached_M/N (-1 means not computed)

    AngularPartition(const Point& c, double start, double end, int id)
        : center(c), angle_start(start), angle_end(end), partition_id(id), boundary_arc(INFINITY),
          cos_start(cos(start)), sin_start(sin(start)),
          cos_end(cos(end)), sin_end(sin(end)),
          cached_arc_radius(-1) {}

    // Update cached M and N when boundary_arc changes
    inline void updateCachedMN() {
        if (cached_arc_radius != boundary_arc) {
            cached_M = Point(center.x + boundary_arc * cos_start,
                            center.y + boundary_arc * sin_start);
            cached_N = Point(center.x + boundary_arc * cos_end,
                            center.y + boundary_arc * sin_end);
            cached_arc_radius = boundary_arc;
        }
    }

    // Check if a point falls within this angular partition
    bool contains_point(const Point& p) const {
        double dx = p.x - center.x;
        double dy = p.y - center.y;

        // Special case: point is at center
        if (dx == 0 && dy == 0) {
            return true;  // Center belongs to all partitions conceptually
        }

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

};

// Calculate both min and max subtended angles between a point x and a partition P
// The subtended angle is measured at the query point q
// Sets is_in_partition, min_angle, and max_angle output parameters
// This avoids duplicate dx, dy, atan2, and angular_distance calculations
inline void computeMinMaxAngles(const Point& query_point, const Point& x, const AngularPartition& partition,
                                bool& is_in_partition, double& min_angle, double& max_angle) {
    // Calculate the angle from query_point to x (computed once)
    double dx = x.x - query_point.x;
    double dy = x.y - query_point.y;
    double angle_to_x = normalize_angle(atan2(dy, dx));

    // Calculate angular distances to both boundaries (computed once, used for both min and max)
    double dist_to_start = angular_distance(angle_to_x, partition.angle_start);
    double dist_to_end = angular_distance(angle_to_x, partition.angle_end);

    // Check if point is in partition
    is_in_partition = partition.contains_point(x);

    // Min angle: 0 if in partition, otherwise smaller of the two distances
    min_angle = is_in_partition ? 0.0 : min(dist_to_start, dist_to_end);

    // Max angle: larger of the two angular distances
    max_angle = max(dist_to_start, dist_to_end);
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
// Parameters dist_fq and is_in_partition are passed to avoid redundant calculations
// Uses cached M/N points from partition (must call updateCachedMN before if boundary_arc changed)
bool isSignificantFacility(const Point& f, double dist_fq, bool is_in_partition, AngularPartition& partition) {
    // If boundary arc is not set (INFINITY), we cannot prune, so keep the facility
    if (partition.boundary_arc == INFINITY) {
        return true;  // Keep as potentially significant when no boundary is set
    }

    // Lemma 1: if f is in P and dist(f,q) > 2 * boundary_arc, f is not significant
    if (is_in_partition) {
        return dist_fq <= 2.0 * partition.boundary_arc;
    }

    // Lemma 2: if f is not in P, check distances to cached M and N
    partition.updateCachedMN();
    double dist_Mf = partition.cached_M.distance_to(f);
    double dist_Nf = partition.cached_N.distance_to(f);

    // f is not significant if both distances are greater than boundary_arc
    return !(dist_Mf > partition.boundary_arc && dist_Nf > partition.boundary_arc);
}

// Check if a node/MBR may contain a significant facility for at least one partition
bool mayContainSignificantFacility(const Rectangle& mbr,
                                   vector<AngularPartition>& partitions,
                                   const Point& query_point) {
    // Get the four corners of the MBR and their distances to query (compute once)
    Point corners[4] = {
        Point(mbr.min_x, mbr.min_y),
        Point(mbr.max_x, mbr.min_y),
        Point(mbr.min_x, mbr.max_y),
        Point(mbr.max_x, mbr.max_y)
    };
    double corner_dists[4];
    int corner_partitions[4];
    for (int i = 0; i < 4; i++) {
        corner_dists[i] = corners[i].distance_to(query_point);
        double dx = corners[i].x - query_point.x;
        double dy = corners[i].y - query_point.y;
        corner_partitions[i] = getPartitionIndex(dx, dy);
    }

    // Calculate min distance from MBR to query point once (doesn't depend on partition)
    double min_dist_mbr_to_q = min_distance_to_rect(query_point, mbr);

    // Check each partition
    for (auto& partition : partitions) {
        // If boundary arc is not set, we can't prune
        if (partition.boundary_arc == INFINITY) {
            return true;
        }

        // Conservative check first (cheapest): if MBR is close enough to query point
        if (min_dist_mbr_to_q <= 2.0 * partition.boundary_arc) {
            return true;
        }

        // Update cached M/N for this partition
        partition.updateCachedMN();

        // Check if any corner could be significant for this partition
        for (int i = 0; i < 4; i++) {
            // Check Lemma 1: if corner is in this partition
            if (corner_partitions[i] == partition.partition_id) {
                // Corner could be significant if dist <= 2 * boundary_arc
                if (corner_dists[i] <= 2.0 * partition.boundary_arc) {
                    return true;  // MBR may contain significant facility
                }
            } else {
                // Check Lemma 2: if corner is not in partition, use cached M and N
                double dist_Mc = partition.cached_M.distance_to(corners[i]);
                double dist_Nc = partition.cached_N.distance_to(corners[i]);

                // If either distance is <= boundary_arc, MBR might contain significant facility
                if (dist_Mc <= partition.boundary_arc || dist_Nc <= partition.boundary_arc) {
                    return true;
                }
            }
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

    // Calculate dist_fq once outside the loop
    double dist_fq = f.distance_to(query_point);

    // Process each partition for this facility
    for (auto& partition : partitions) {
        bool is_in_partition;
        double min_ang, max_ang;
        computeMinMaxAngles(query_point, f, partition, is_in_partition, min_ang, max_ang);

        // Only consider this facility if minAngle < 90 degrees
        if (min_ang < NINETY_DEGREES) {

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
            if ((int)partition.upper_bounds_heap.size() < k) {
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

            // Check if f is significant using lemmas (pass pre-computed values)
            if (isSignificantFacility(f, dist_fq, is_in_partition, partition)) {
                // Calculate lower bound
                double lower_bound;
                if (is_in_partition) {
                    // f is in the partition, lower bound is just half distance
                    lower_bound = dist_fq / 2.0;
                } else {
                    lower_bound = dist_fq / (2.0 * cos(min_ang));
                }

                // Add to sigList in sorted position (by lower bound)
                FacilityBound new_entry(f, lower_bound, upper_bound);
                auto insert_pos = std::lower_bound(partition.sigList.begin(), partition.sigList.end(), new_entry);
                partition.sigList.insert(insert_pos, new_entry);
            }
        }
    }
}

// Helper function to get partition index using relative positions
// Avoids expensive trigonometric calculations
inline int getPartitionIndex(double dx, double dy) {
    // Handle special cases
    if (dx == 0 && dy == 0) return 0;  // Point at query location

    int partition_idx = 0;

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
bool isRkNN(const Point& u, double dist_uq, double dist_uq_sq,
            const AngularPartition& user_partition, int k) {
    // This function now takes only the specific partition that u lies in
    // and the pre-calculated distance from u to q (both regular and squared)
    int count = 0;

    // Check facilities in sigList in ascending order of lower bound
    for (const auto& fb : user_partition.sigList) {
        // If dist(u,q) <= lower bound of f to P, u is RkNN
        if (dist_uq <= fb.lower_bound) {
            return true;
        }

        // Check if this facility is closer to u than q is (use squared distance to avoid sqrt)
        double dx = u.x - fb.facility.x;
        double dy = u.y - fb.facility.y;
        double dist_uf_sq = dx * dx + dy * dy;
        if (dist_uf_sq < dist_uq_sq) {
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
            // Directly calculate which partition this user belongs to using relative positions
            double dx = entry.point.x - query_point.x;
            double dy = entry.point.y - query_point.y;

            int partition_idx = getPartitionIndex(dx, dy);
            const auto& partition = partitions[partition_idx];

            // Calculate squared distance first (reuse dx, dy from above)
            double dist_sq = dx * dx + dy * dy;

            // Check if user is in pruned area of this partition (use squared comparison)
            double boundary_arc_sq = partition.boundary_arc * partition.boundary_arc;
            if (dist_sq > boundary_arc_sq) {
                continue;  // Pruned, skip to next entry
            }

            // Only compute sqrt when needed (not pruned)
            double dist_to_q = sqrt(dist_sq);

            // Verify if this user is RkNN
            // Pass the partition, distance, and squared distance
            if (isRkNN(entry.point, dist_to_q, dist_sq, partition, k)) {
                rknn_results.push_back(entry.point);
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
