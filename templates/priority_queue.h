

#ifndef PRIORITY_QUEUE_H
#define PRIORITY_QUEUE_H

// struct heap_element {
// 	int id;
// 	long priority;
// };

// struct PriorityQueue {
// 	struct heap_element* array;
// 	unsigned size;
// 	unsigned max_size;
// };

struct PriorityQueue* new_heap(unsigned const heap_size);

int heap_insert(struct PriorityQueue* const heap, const struct heap_element element);

struct heap_element* find_heap_min(const struct PriorityQueue* heap);

int delete_heap_min(struct PriorityQueue* const heap);

void free_heap(struct PriorityQueue* const heap);

#endif
