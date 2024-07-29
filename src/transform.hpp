#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

#include <ostream>

/*
TODO: Use matrices instead
- multiplication
- inverse
*/

struct Transform{
    double xOffset;
    double yOffset;
    double scale;
    double rotation;
    double response;

    inline Transform operator+(const Transform& transform) const{
        return Transform{
            .xOffset = xOffset + transform.xOffset,
            .yOffset = yOffset + transform.yOffset,
            .scale = (scale + transform.scale) * 0.5, // TODO: This is totally arbitrary
            .rotation = rotation + transform.rotation,
            .response = (response + transform.response) * 0.5, // TODO: This is totally arbitrary
        };
    }

    inline Transform operator-(const Transform& transform) const{
        return Transform{
            .xOffset = xOffset - transform.xOffset,
            .yOffset = yOffset - transform.yOffset,
            .scale = (scale + transform.scale) * 0.5,
            .rotation = rotation - transform.rotation,
            .response = (response + transform.response) * 0.5,
        };
    }

    inline Transform& operator+=(const Transform& transform){
        *this = *this + transform;
        return *this;
    }

    inline Transform& operator-=(const Transform& transform){
        *this = *this - transform;
        return *this;
    }
};

inline std::ostream& operator<<(std::ostream& os, const Transform& t){
    os << "Transform(" << t.xOffset << ", " << t.yOffset << ", " << t.scale << ", " << t.rotation << ", " << t.response << ")";
    return os;
}

#endif // __TRANSFORM_H__