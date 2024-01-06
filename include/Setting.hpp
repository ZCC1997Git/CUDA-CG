#ifndef _SETTING_
#define _SETTING_
#include <Datatype.hpp>
/*the number of nodes in each x and y directions*/
inline constexpr unsigned int Xnode = 2048;
/*the number of nodes in each x and y directions*/
inline constexpr unsigned int Ynode = Xnode;

/*the number of nodes in each x and y directions*/
inline constexpr type_t Q = 1.0;

/*the range of the problem*/
inline constexpr type_t R = 1.0;

/*space step*/
inline constexpr type_t h = R / (Xnode - 1);

/*the 2D pointer used to store the Martix*/
inline type_t** A = nullptr;

/*the 1D pointer used to store the vector x*/
inline type_t* x = nullptr;
inline type_t* x_all = nullptr;

/*the 1D pointer used to store the vector b*/
inline type_t* b = nullptr;

#endif