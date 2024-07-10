#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

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

#endif // __TRANSFORM_H__