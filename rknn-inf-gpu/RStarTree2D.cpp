#include "RStarTree2D.h"

// Point method implementations
double Point::distance_to(const Point& other) const {
    return sqrt(pow(x - other.x, 2) + pow(y - other.y, 2));
}

bool Point::operator<(const Point& other) const {
    if (x != other.x) return x < other.x;
    if (y != other.y) return y < other.y;
    return id < other.id;
}

Rectangle Point::to_rectangle() const {
    return Rectangle(x, y, x, y);
}

// Rectangle method implementations
double Rectangle::area() const {
    return (max_x - min_x) * (max_y - min_y);
}

double Rectangle::perimeter() const {
    return 2 * ((max_x - min_x) + (max_y - min_y));
}

bool Rectangle::intersects(const Rectangle& other) const {
    return !(max_x < other.min_x || min_x > other.max_x ||
            max_y < other.min_y || min_y > other.max_y);
}

bool Rectangle::contains(const Rectangle& other) const {
    return min_x <= other.min_x && min_y <= other.min_y &&
           max_x >= other.max_x && max_y >= other.max_y;
}

bool Rectangle::contains_point(const Point& point) const {
    return point.x >= min_x && point.x <= max_x &&
           point.y >= min_y && point.y <= max_y;
}

Rectangle Rectangle::union_with(const Rectangle& other) const {
    return Rectangle(
        min(min_x, other.min_x),
        min(min_y, other.min_y),
        max(max_x, other.max_x),
        max(max_y, other.max_y)
    );
}

Rectangle Rectangle::union_with_point(const Point& point) const {
    return Rectangle(
        min(min_x, point.x),
        min(min_y, point.y),
        max(max_x, point.x),
        max(max_y, point.y)
    );
}

double Rectangle::enlargement_area(const Rectangle& other) const {
    Rectangle united = union_with(other);
    return united.area() - area();
}

double Rectangle::enlargement_area_point(const Point& point) const {
    Rectangle united = union_with_point(point);
    return united.area() - area();
}

// RStarEntry method implementations
RStarEntry::RStarEntry(const Point& pt) : mbr(pt.to_rectangle()), child(nullptr), point(pt) {}

RStarEntry::RStarEntry(const Rectangle& rect, shared_ptr<RStarNode> node)
    : mbr(rect), child(node), point() {}

bool RStarEntry::is_leaf_entry() const {
    return child == nullptr;
}

// RStarNode method implementations
RStarNode::RStarNode(bool leaf) : is_leaf(leaf), parent(nullptr) {}

Rectangle RStarNode::get_mbr() const {
    if (entries.empty()) {
        return Rectangle();
    }

    Rectangle mbr = entries[0].mbr;
    for (size_t i = 1; i < entries.size(); ++i) {
        mbr = mbr.union_with(entries[i].mbr);
    }
    return mbr;
}

bool RStarNode::is_full() const {
    return entries.size() >= MAX_ENTRIES;
}

bool RStarNode::is_underflow() const {
    return entries.size() < MIN_ENTRIES;
}

// RStarTree method implementations
RStarTree::RStarTree() : root(make_shared<RStarNode>(true)), reinsert_count(0) {}

void RStarTree::insert(const Point& point) {
    RStarEntry entry(point);
    insert_entry(entry, -1);  // -1 means "traverse to leaves"
}

void RStarTree::insert(double x, double y, int id) {
    Point point(x, y, id);
    insert(point);
}

vector<Point> RStarTree::range_search(const Rectangle& query_rect) {
    vector<Point> results;
    range_search_recursive(root, query_rect, results);
    return results;
}

vector<Point> RStarTree::range_search(double min_x, double min_y, double max_x, double max_y) {
    Rectangle query_rect(min_x, min_y, max_x, max_y);
    return range_search(query_rect);
}

vector<Point> RStarTree::radius_search(const Point& query_point, double radius) {
    Rectangle search_rect(
        query_point.x - radius, query_point.y - radius,
        query_point.x + radius, query_point.y + radius
    );

    vector<Point> candidates = range_search(search_rect);
    vector<Point> results;

    for (const auto& candidate : candidates) {
        if (query_point.distance_to(candidate) <= radius) {
            results.push_back(candidate);
        }
    }

    return results;
}

vector<Point> RStarTree::knn_search(const Point& query_point, int k) {
    priority_queue<pair<double, Point>> pq;  // Max heap by distance
    knn_search_recursive(root, query_point, k, pq);

    vector<Point> results;
    while (!pq.empty()) {
        results.push_back(pq.top().second);
        pq.pop();
    }

    reverse(results.begin(), results.end());  // Closest first
    return results;
}

bool RStarTree::remove(const Point& point) {
    return remove_point(root, point);
}

bool RStarTree::remove(double x, double y, int id) {
    Point point(x, y, id);
    return remove(point);
}

void RStarTree::print_tree() {
    print_node(root, 0);
}

void RStarTree::insert_entry(RStarEntry& entry, int level) {
    auto target_node = choose_subtree(root, entry, level);

    if (!target_node->is_full()) {
        target_node->entries.push_back(entry);
        adjust_tree(target_node);
    } else {
        // Overflow treatment
        if (level != 0 && reinsert_count == 0) {
            reinsert_count++;
            reinsert(target_node, entry);
            reinsert_count = 0;
        } else {
            auto new_node = split_node(target_node, entry);
            if (target_node == root) {
                // Create new root
                auto new_root = make_shared<RStarNode>(false);
                new_root->entries.emplace_back(target_node->get_mbr(), target_node);
                new_root->entries.emplace_back(new_node->get_mbr(), new_node);
                target_node->parent = new_root;
                new_node->parent = new_root;
                root = new_root;
            } else {
                // Insert new node into parent
                RStarEntry new_entry(new_node->get_mbr(), new_node);
                insert_into_parent(target_node->parent, new_entry);
            }
        }
    }
}

shared_ptr<RStarNode> RStarTree::choose_subtree(shared_ptr<RStarNode> node,
                                                     const RStarEntry& entry,
                                                     int target_level,
                                                     int current_level) {
    // Special case: target_level = -1 means "traverse to leaves"
    if (target_level == -1) {
        if (node->is_leaf) {
            return node;
        }
    } else {
        // Stop when we reach target level OR hit a leaf
        if (current_level == target_level || node->is_leaf) {
            return node;
        }
    }

    // Choose subtree with minimum overlap enlargement, then minimum area enlargement
    shared_ptr<RStarNode> best_child = nullptr;
    double min_overlap_enlargement = numeric_limits<double>::max();
    double min_area_enlargement = numeric_limits<double>::max();
    double min_area = numeric_limits<double>::max();

    for (auto& child_entry : node->entries) {
        double area_enlargement, overlap_enlargement;

        if (entry.is_leaf_entry()) {
            // For point entries, use point-based calculations
            area_enlargement = child_entry.mbr.enlargement_area_point(entry.point);
            overlap_enlargement = calculate_overlap_enlargement_point(node, child_entry, entry.point);
        } else {
            // For child node entries, use rectangle-based calculations
            area_enlargement = child_entry.mbr.enlargement_area(entry.mbr);
            overlap_enlargement = calculate_overlap_enlargement_rect(node, child_entry, entry.mbr);
        }

        double area = child_entry.mbr.area();

        bool better = false;
        if (overlap_enlargement < min_overlap_enlargement) {
            better = true;
        } else if (overlap_enlargement == min_overlap_enlargement) {
            if (area_enlargement < min_area_enlargement) {
                better = true;
            } else if (area_enlargement == min_area_enlargement && area < min_area) {
                better = true;
            }
        }

        if (better) {
            min_overlap_enlargement = overlap_enlargement;
            min_area_enlargement = area_enlargement;
            min_area = area;
            best_child = child_entry.child;
        }
    }

    return choose_subtree(best_child, entry, target_level, current_level + 1);
}

shared_ptr<RStarNode> RStarTree::choose_leaf(shared_ptr<RStarNode> node,
                                                  const Point& point, int target_level) {
    RStarEntry entry(point);
    return choose_subtree(node, entry, target_level);
}

double RStarTree::calculate_overlap_enlargement_point(shared_ptr<RStarNode> node,
                                                     const RStarEntry& entry, const Point& point) {
    Rectangle old_mbr = entry.mbr;
    Rectangle new_mbr = entry.mbr.union_with_point(point);

    double old_overlap = 0, new_overlap = 0;

    for (auto& other_entry : node->entries) {
        if (&other_entry == &entry) continue;

        // Calculate old overlap
        if (old_mbr.intersects(other_entry.mbr)) {
            Rectangle intersection(
                max(old_mbr.min_x, other_entry.mbr.min_x),
                max(old_mbr.min_y, other_entry.mbr.min_y),
                min(old_mbr.max_x, other_entry.mbr.max_x),
                min(old_mbr.max_y, other_entry.mbr.max_y)
            );
            old_overlap += intersection.area();
        }

        // Calculate new overlap
        if (new_mbr.intersects(other_entry.mbr)) {
            Rectangle intersection(
                max(new_mbr.min_x, other_entry.mbr.min_x),
                max(new_mbr.min_y, other_entry.mbr.min_y),
                min(new_mbr.max_x, other_entry.mbr.max_x),
                min(new_mbr.max_y, other_entry.mbr.max_y)
            );
            new_overlap += intersection.area();
        }
    }

    return new_overlap - old_overlap;
}

double RStarTree::calculate_overlap_enlargement_rect(shared_ptr<RStarNode> node,
                                                    const RStarEntry& entry, const Rectangle& rect) {
    Rectangle old_mbr = entry.mbr;
    Rectangle new_mbr = entry.mbr.union_with(rect);

    double old_overlap = 0, new_overlap = 0;

    for (auto& other_entry : node->entries) {
        if (&other_entry == &entry) continue;

        // Calculate old overlap
        if (old_mbr.intersects(other_entry.mbr)) {
            Rectangle intersection(
                max(old_mbr.min_x, other_entry.mbr.min_x),
                max(old_mbr.min_y, other_entry.mbr.min_y),
                min(old_mbr.max_x, other_entry.mbr.max_x),
                min(old_mbr.max_y, other_entry.mbr.max_y)
            );
            old_overlap += intersection.area();
        }

        // Calculate new overlap
        if (new_mbr.intersects(other_entry.mbr)) {
            Rectangle intersection(
                max(new_mbr.min_x, other_entry.mbr.min_x),
                max(new_mbr.min_y, other_entry.mbr.min_y),
                min(new_mbr.max_x, other_entry.mbr.max_x),
                min(new_mbr.max_y, other_entry.mbr.max_y)
            );
            new_overlap += intersection.area();
        }
    }

    return new_overlap - old_overlap;
}

int RStarTree::get_node_level(shared_ptr<RStarNode> node) {
    if (node == root) {
        return 0;  // Root is at level 0
    }

    int level = 0;
    shared_ptr<RStarNode> current = node;

    // Traverse up to root, counting levels
    while (current->parent != nullptr) {
        level++;
        current = current->parent;
    }

    return level;
}

void RStarTree::reinsert(shared_ptr<RStarNode> node, RStarEntry& new_entry) {
    node->entries.push_back(new_entry);

    // Calculate distances from center
    Rectangle node_mbr = node->get_mbr();
    double center_x = (node_mbr.min_x + node_mbr.max_x) / 2.0;
    double center_y = (node_mbr.min_y + node_mbr.max_y) / 2.0;

    vector<pair<double, int>> distances;
    for (size_t i = 0; i < node->entries.size(); ++i) {
        double entry_center_x, entry_center_y;
        if (node->is_leaf) {
            entry_center_x = node->entries[i].point.x;
            entry_center_y = node->entries[i].point.y;
        } else {
            Rectangle& rect = node->entries[i].mbr;
            entry_center_x = (rect.min_x + rect.max_x) / 2.0;
            entry_center_y = (rect.min_y + rect.max_y) / 2.0;
        }

        double distance = sqrt(pow(entry_center_x - center_x, 2) +
                                  pow(entry_center_y - center_y, 2));
        distances.emplace_back(distance, i);
    }

    // Sort by distance (farthest first)
    sort(distances.begin(), distances.end(), greater<pair<double, int>>());

    // Remove entries for reinsertion (30% of max entries)
    int num_reinsert = max(1, (int)(RStarNode::MAX_ENTRIES * REINSERT_FACTOR / 100));
    vector<RStarEntry> reinsert_entries;

    for (int i = 0; i < num_reinsert && !distances.empty(); ++i) {
        int idx = distances[i].second;
        reinsert_entries.push_back(node->entries[idx]);
    }

    // Remove entries from node (in reverse order to maintain indices)
    vector<int> indices_to_remove;
    for (int i = 0; i < num_reinsert && !distances.empty(); ++i) {
        indices_to_remove.push_back(distances[i].second);
    }
    sort(indices_to_remove.rbegin(), indices_to_remove.rend());

    for (int idx : indices_to_remove) {
        node->entries.erase(node->entries.begin() + idx);
    }

    adjust_tree(node);

    // Reinsert entries at their original level
    int original_level = get_node_level(node);
    for (auto& entry : reinsert_entries) {
        if (node->is_leaf) {
            // Point entries must traverse to actual leaf nodes
            insert_entry(entry, -1);  // -1 means "traverse to leaves"
        } else {
            // Child node entries go back to the same internal level they came from
            insert_entry(entry, original_level);
        }
    }
}

shared_ptr<RStarNode> RStarTree::split_node(shared_ptr<RStarNode> node, RStarEntry& new_entry) {
    node->entries.push_back(new_entry);

    // R* split algorithm
    int split_axis = choose_split_axis(node);
    int split_index = choose_split_index(node, split_axis);

    auto new_node = make_shared<RStarNode>(node->is_leaf);
    new_node->parent = node->parent;

    // Sort entries along the chosen axis
    sort_entries(node, split_axis);

    // Split entries
    for (size_t i = split_index; i < node->entries.size(); ++i) {
        new_node->entries.push_back(node->entries[i]);
        if (!node->is_leaf && node->entries[i].child) {
            node->entries[i].child->parent = new_node;
        }
    }

    node->entries.erase(node->entries.begin() + split_index, node->entries.end());

    return new_node;
}

int RStarTree::choose_split_axis(shared_ptr<RStarNode> node) {
    double x_goodness = calculate_axis_goodness(node, 0);  // x-axis
    double y_goodness = calculate_axis_goodness(node, 1);  // y-axis

    return (x_goodness <= y_goodness) ? 0 : 1;
}

double RStarTree::calculate_axis_goodness(shared_ptr<RStarNode> node, int axis) {
    sort_entries(node, axis);

    double total_margin = 0;
    int min_entries = RStarNode::MIN_ENTRIES;
    int max_split = node->entries.size() - min_entries;

    for (int k = min_entries; k <= max_split; ++k) {
        Rectangle left_mbr = get_mbr_for_entries(node, 0, k);
        Rectangle right_mbr = get_mbr_for_entries(node, k, node->entries.size());
        total_margin += left_mbr.perimeter() + right_mbr.perimeter();
    }

    return total_margin;
}

int RStarTree::choose_split_index(shared_ptr<RStarNode> node, int axis) {
    sort_entries(node, axis);

    int best_index = RStarNode::MIN_ENTRIES;
    double min_overlap = numeric_limits<double>::max();
    double min_area = numeric_limits<double>::max();

    int min_entries = RStarNode::MIN_ENTRIES;
    int max_split = node->entries.size() - min_entries;

    for (int k = min_entries; k <= max_split; ++k) {
        Rectangle left_mbr = get_mbr_for_entries(node, 0, k);
        Rectangle right_mbr = get_mbr_for_entries(node, k, node->entries.size());

        double overlap = 0;
        if (left_mbr.intersects(right_mbr)) {
            Rectangle intersection(
                max(left_mbr.min_x, right_mbr.min_x),
                max(left_mbr.min_y, right_mbr.min_y),
                min(left_mbr.max_x, right_mbr.max_x),
                min(left_mbr.max_y, right_mbr.max_y)
            );
            overlap = intersection.area();
        }

        double total_area = left_mbr.area() + right_mbr.area();

        if (overlap < min_overlap || (overlap == min_overlap && total_area < min_area)) {
            min_overlap = overlap;
            min_area = total_area;
            best_index = k;
        }
    }

    return best_index;
}

void RStarTree::sort_entries(shared_ptr<RStarNode> node, int axis) {
    if (axis == 0) {  // x-axis
        if (node->is_leaf) {
            sort(node->entries.begin(), node->entries.end(),
                     [](const RStarEntry& a, const RStarEntry& b) {
                         return a.point.x < b.point.x;
                     });
        } else {
            sort(node->entries.begin(), node->entries.end(),
                     [](const RStarEntry& a, const RStarEntry& b) {
                         return a.mbr.min_x < b.mbr.min_x;
                     });
        }
    } else {  // y-axis
        if (node->is_leaf) {
            sort(node->entries.begin(), node->entries.end(),
                     [](const RStarEntry& a, const RStarEntry& b) {
                         return a.point.y < b.point.y;
                     });
        } else {
            sort(node->entries.begin(), node->entries.end(),
                     [](const RStarEntry& a, const RStarEntry& b) {
                         return a.mbr.min_y < b.mbr.min_y;
                     });
        }
    }
}

Rectangle RStarTree::get_mbr_for_entries(shared_ptr<RStarNode> node, int start, int end) {
    if (start >= end) return Rectangle();

    Rectangle mbr = node->entries[start].mbr;
    for (int i = start + 1; i < end; ++i) {
        mbr = mbr.union_with(node->entries[i].mbr);
    }
    return mbr;
}

void RStarTree::insert_into_parent(shared_ptr<RStarNode> parent, RStarEntry& entry) {
    if (!parent->is_full()) {
        parent->entries.push_back(entry);
        entry.child->parent = parent;
        adjust_tree(parent);
    } else {
        auto new_node = split_node(parent, entry);
        if (parent == root) {
            auto new_root = make_shared<RStarNode>(false);
            new_root->entries.emplace_back(parent->get_mbr(), parent);
            new_root->entries.emplace_back(new_node->get_mbr(), new_node);
            parent->parent = new_root;
            new_node->parent = new_root;
            root = new_root;
        } else {
            RStarEntry new_entry(new_node->get_mbr(), new_node);
            insert_into_parent(parent->parent, new_entry);
        }
    }
}

void RStarTree::adjust_tree(shared_ptr<RStarNode> node) {
    while (node != root && node->parent) {
        // Update parent's MBR
        for (auto& entry : node->parent->entries) {
            if (entry.child == node) {
                entry.mbr = node->get_mbr();
                break;
            }
        }
        node = node->parent;
    }
}

void RStarTree::range_search_recursive(shared_ptr<RStarNode> node, const Rectangle& query_rect,
                                      vector<Point>& results) {
    if (node->is_leaf) {
        for (const auto& entry : node->entries) {
            if (query_rect.contains_point(entry.point)) {
                results.push_back(entry.point);
            }
        }
    } else {
        for (const auto& entry : node->entries) {
            if (query_rect.intersects(entry.mbr)) {
                range_search_recursive(entry.child, query_rect, results);
            }
        }
    }
}

void RStarTree::knn_search_recursive(shared_ptr<RStarNode> node, const Point& query_point,
                                    int k, priority_queue<pair<double, Point>>& pq) {
    if (node->is_leaf) {
        for (const auto& entry : node->entries) {
            // Skip the query point itself based on ID
            if (entry.point.id == query_point.id) {
                continue;
            }

            double distance = query_point.distance_to(entry.point);

            if (static_cast<int>(pq.size()) < k) {
                pq.emplace(distance, entry.point);
            } else if (distance < pq.top().first) {
                pq.pop();
                pq.emplace(distance, entry.point);
            }
        }
    } else {
        // Create priority queue for child nodes based on minimum distance to MBR
        priority_queue<pair<double, shared_ptr<RStarNode>>,
                          vector<pair<double, shared_ptr<RStarNode>>>,
                          greater<pair<double, shared_ptr<RStarNode>>>> child_pq;

        for (const auto& entry : node->entries) {
            double min_dist = min_distance_to_rect(query_point, entry.mbr);
            child_pq.emplace(min_dist, entry.child);
        }

        while (!child_pq.empty()) {
            auto current = child_pq.top();
            child_pq.pop();

            // Prune if we have k points and this node can't contain closer points
            if (static_cast<int>(pq.size()) == k && current.first > pq.top().first) {
                break;
            }

            knn_search_recursive(current.second, query_point, k, pq);
        }
    }
}

double RStarTree::min_distance_to_rect(const Point& point, const Rectangle& rect) {
    double dx = max({rect.min_x - point.x, 0.0, point.x - rect.max_x});
    double dy = max({rect.min_y - point.y, 0.0, point.y - rect.max_y});
    return sqrt(dx * dx + dy * dy);
}

bool RStarTree::remove_point(shared_ptr<RStarNode> node, const Point& point) {
    if (node->is_leaf) {
        for (auto it = node->entries.begin(); it != node->entries.end(); ++it) {
            if (it->point.x == point.x && it->point.y == point.y && it->point.id == point.id) {
                node->entries.erase(it);

                if (node->is_underflow() && node != root) {
                    handle_underflow(node);
                } else {
                    adjust_tree(node);
                }
                return true;
            }
        }
    } else {
        for (const auto& entry : node->entries) {
            if (entry.mbr.contains_point(point)) {
                if (remove_point(entry.child, point)) {
                    return true;
                }
            }
        }
    }
    return false;
}

void RStarTree::handle_underflow(shared_ptr<RStarNode> node) {
    // Simplified underflow handling
    if (node->parent && node->parent->entries.size() > 1) {
        adjust_tree(node);
    }
}

void RStarTree::print_node(shared_ptr<RStarNode> node, int depth) {
    string indent(depth * 2, ' ');
    cout << indent << (node->is_leaf ? "Leaf" : "Internal") << " node with "
              << node->entries.size() << " entries:" << endl;

    for (const auto& entry : node->entries) {
        cout << indent << "  MBR: (" << entry.mbr.min_x << "," << entry.mbr.min_y
                  << ")-(" << entry.mbr.max_x << "," << entry.mbr.max_y << ")";
        if (entry.is_leaf_entry()) {
            cout << " Point: (" << entry.point.x << "," << entry.point.y
                      << ") ID: " << entry.point.id;
        }
        cout << endl;

        if (!entry.is_leaf_entry()) {
            print_node(entry.child, depth + 1);
        }
    }
}