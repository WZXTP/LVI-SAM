#include "CameraFactory.h"

#include <boost/algorithm/string.hpp>


#include "CataCamera.h"
#include "EquidistantCamera.h"
#include "PinholeCamera.h"
#include "ScaramuzzaCamera.h"

#include "ceres/ceres.h"

namespace camodocal
{

boost::shared_ptr<CameraFactory> CameraFactory::m_instance;

CameraFactory::CameraFactory()
{

}

boost::shared_ptr<CameraFactory>
CameraFactory::instance(void)
{
    if (m_instance.get() == 0)
    {
        m_instance.reset(new CameraFactory);
    }

    return m_instance;
}

CameraPtr
//根据相机的模型类型（modelType）和其他参数（如相机名称、图像尺寸）生成相应的相机对象，并设置相应的参数。
CameraFactory::generateCamera(Camera::ModelType modelType,
                              const std::string& cameraName,
                              cv::Size imageSize) const
{
    switch (modelType)
    {
    case Camera::KANNALA_BRANDT://KANNALA_BRANDT：生成 EquidistantCamera 对象。
    {
        EquidistantCameraPtr camera(new EquidistantCamera);

        EquidistantCamera::Parameters params = camera->getParameters();
        params.cameraName() = cameraName;
        params.imageWidth() = imageSize.width;
        params.imageHeight() = imageSize.height;
        camera->setParameters(params);
        return camera;
    }
    case Camera::PINHOLE://PINHOLE：生成 PinholeCamera 对象。
    {
        PinholeCameraPtr camera(new PinholeCamera);

        PinholeCamera::Parameters params = camera->getParameters();
        params.cameraName() = cameraName;
        params.imageWidth() = imageSize.width;
        params.imageHeight() = imageSize.height;
        camera->setParameters(params);
        return camera;
    }
    case Camera::SCARAMUZZA://SCARAMUZZA：生成 OCAMCamera 对象。
    {
        OCAMCameraPtr camera(new OCAMCamera);

        OCAMCamera::Parameters params = camera->getParameters();
        params.cameraName() = cameraName;
        params.imageWidth() = imageSize.width;
        params.imageHeight() = imageSize.height;
        camera->setParameters(params);
        return camera;
    }
    case Camera::MEI://MEI 或默认类型：生成 CataCamera 对象。
    default:
    {
        CataCameraPtr camera(new CataCamera);

        CataCamera::Parameters params = camera->getParameters();
        params.cameraName() = cameraName;
        params.imageWidth() = imageSize.width;
        params.imageHeight() = imageSize.height;
        camera->setParameters(params);
        return camera;
    }
    }
}

CameraPtr
CameraFactory::generateCameraFromYamlFile(const std::string& filename)//该方法从YAML文件中读取相机的配置，并生成相应的相机对象。
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        return CameraPtr();
    }

    Camera::ModelType modelType = Camera::MEI;
    if (!fs["model_type"].isNone())
    {
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if (boost::iequals(sModelType, "kannala_brandt"))
        {
            modelType = Camera::KANNALA_BRANDT;
        }
        else if (boost::iequals(sModelType, "mei"))
        {
            modelType = Camera::MEI;
        }
        else if (boost::iequals(sModelType, "scaramuzza"))
        {
            modelType = Camera::SCARAMUZZA;
        }
        else if (boost::iequals(sModelType, "pinhole"))
        {
            modelType = Camera::PINHOLE;
        }
        else
        {
            std::cerr << "# ERROR: Unknown camera model: " << sModelType << std::endl;
            return CameraPtr();
        }
    }

    switch (modelType)
    {
    case Camera::KANNALA_BRANDT:
    {
        EquidistantCameraPtr camera(new EquidistantCamera);

        EquidistantCamera::Parameters params = camera->getParameters();
        params.readFromYamlFile(filename);
        camera->setParameters(params);
        return camera;
    }
    case Camera::PINHOLE:
    {
        PinholeCameraPtr camera(new PinholeCamera);

        PinholeCamera::Parameters params = camera->getParameters();
        params.readFromYamlFile(filename);
        camera->setParameters(params);
        return camera;
    }
    case Camera::SCARAMUZZA:
    {
        OCAMCameraPtr camera(new OCAMCamera);

        OCAMCamera::Parameters params = camera->getParameters();
        params.readFromYamlFile(filename);
        camera->setParameters(params);
        return camera;
    }
    case Camera::MEI:
    default:
    {
        CataCameraPtr camera(new CataCamera);

        CataCamera::Parameters params = camera->getParameters();
        params.readFromYamlFile(filename);
        camera->setParameters(params);
        return camera;
    }
    }

    return CameraPtr();
}

}

