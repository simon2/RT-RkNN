// define a structure to represent a point in 2D space with an ID
typedef struct {
    uint32_t id;
    uint32_t x, y;
} Point;

float cal_dist(Point a, Point b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}