
#ifndef PRIORITY_QUEUE_INTERFACE_H
#define PRIORITY_QUEUE_INTERFACE_H

#include "priority_queue.h"



struct heap_element Node(int id, long priority);

int tuple(int* coordinates);

int* np_array(int coordinate);

unsigned len(struct PriorityQueue* queue);

void PriorityQueue_push(struct PriorityQueue* queue, struct heap_element element);

struct heap_element PriorityQueue_pop(struct PriorityQueue* queue);

#endif
