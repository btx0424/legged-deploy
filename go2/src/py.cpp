#include <string>
#include <thread>

#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>
#include "unitree/idl/go2/LowState_.hpp"
#include <unitree/idl/go2/SportModeState_.hpp>
#include "unitree/idl/go2/LowCmd_.hpp"
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"

#include "robot_interface.hpp"
#include "gamepad.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/chrono.h"
#include "pybind11/numpy.h"

using namespace unitree::common;
using namespace unitree::robot;
namespace py = pybind11;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_HIGHSTATE "rt/sportmodestate"

struct RobotState {
    std::array<float, 12> jpos;
    std::array<float, 12> jpos_des;
    std::array<float, 12> jvel;
    std::array<float, 12> jvel_raw;
    std::array<float, 12> jacc;
    std::array<float, 12> tau_est;

    std::array<float, 4> quat;
    std::array<float, 3> gyro;
    std::array<float, 3> rpy;
    std::array<float, 3> acc;
    std::array<int16_t, 4> foot_force;

    double time_since_state_update, state_update_interval;
    double time_since_control_update;
    double time_since_control_application;
    double control_application_since_state_update;
    int control_mode;
};

void lerp_(std::array<float, 12>& a, const std::array<float, 12>& b, float t) {
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = a[i] + t * (b[i] - a[i]);
    }
}

// class SecondOrderLowPassFilter {
//     public:
//         SecondOrderLowPassFilter(float cutoff, float fs) {
//             // Compute the filter coefficients (second-order, low-pass)
//             float nyquist = 0.5 * fs;
//             float normal_cutoff = cutoff / nyquist;
//             // b = {1.0, -2.0 * std::cos(normal_cutoff), 1.0};
//             // a = {1.0, -2.0 * std::cos(normal_cutoff), 1.0};
//             b = {1.0f, -2.0f * static_cast<float>(std::cos(normal_cutoff)), 1.0f};
//             a = {1.0f, -2.0f * static_cast<float>(std::cos(normal_cutoff)), 1.0f};


//             for (int i = 0; i < 12; i++) {
//                 x_tm1[i] = 0.0;
//                 x_tm2[i] = 0.0;
//                 y_tm1[i] = 0.0;
//                 y_tm2[i] = 0.0;
//             };
//         }

//         void Update(const std::array<float, 12>& x) {
//             for (int i = 0; i < 12; i++) {
//                 y[i] = b[0] * x[i] + b[1] * x_tm1[i] + b[2] * x_tm2[i] - a[1] * y_tm1[i] - a[2] * y_tm2[i];
//                 x_tm2[i] = x_tm1[i];
//                 x_tm1[i] = x[i];
//                 y_tm2[i] = y_tm1[i];
//                 y_tm1[i] = y[i];
//             }
//         }
    
//     private:
//         std::array<float, 3> b = {0.293, 0.586, 0.293};
//         std::array<float, 3> a = {1.0, 0.0, 0.172};
//         std::array<float, 12> x_tm1;
//         std::array<float, 12> x_tm2; 
//         std::array<float, 12> y_tm1;
//         std::array<float, 12> y_tm2;
//         std::array<float, 12> y;
// };


class RobotIface
{
public:
    RobotIface() {
        // sport_client.SetTimeout(10.0f);
        // sport_client.Init();
        rsc.SetTimeout(10.0f);
        rsc.Init();

        lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
        lowcmd_publisher->InitChannel();

        lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
        lowstate_subscriber->InitChannel(std::bind(&RobotIface::LowStateMessageHandler, this, std::placeholders::_1), 1);
        
        highstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(TOPIC_HIGHSTATE));
        highstate_subscriber->InitChannel(std::bind(&RobotIface::HighStateMessageHandler, this, std::placeholders::_1), 1);
        
        std::cout << "Init complete!" << std::endl;
    }

    void StartControl(uint64_t interval = 2000) {
        // waiting for gamepad command to start the control thread
        std::chrono::milliseconds duration(100);
        // listen to gamepad command

        // while (true)
        // {
        //     std::cout << "Press R2 to start!" << std::endl;
        //     std::this_thread::sleep_for(duration);

        //     if (gamepad.R2.on_press)
        //     {
        //         break;
        //     }
        // }

        rsc.ServiceSwitch("sport_mode", 1, status);
        // UserControlCallback();

        main_thread_ptr = CreateRecurrentThreadEx("main", UT_CPU_ID_NONE, interval, &RobotIface::MainThreadStep, this);
        
        std::cout << "Start control!" << std::endl;
    }

    void SetCommand(const py::array_t<float> &input_cmd) {
        std::lock_guard<std::mutex> lock(cmd_mutex);
        py::buffer_info buf = input_cmd.request();
        if (buf.size != robot_interface.jpos_des.size()) {
            throw std::runtime_error("Command size must be 12");
        }
        
        std::copy(jpos_des.begin(), jpos_des.begin() + jpos_des.size(), jpos_des_prev.begin());
        std::copy(input_cmd.data(), input_cmd.data() + input_cmd.size(), jpos_des.begin());

        timestamp_command = std::chrono::high_resolution_clock::now();
    }

    py::array_t<float> GetJointPos() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.jpos.size(), robot_interface.jpos.data());
    }

    py::array_t<float> GetJointPosTarget() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.jpos_des.size(), robot_interface.jpos_des.data());
    }

    py::array_t<float> GetJointVel() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.jvel.size(), robot_interface.jvel.data());
    }

    py::array_t<float> GetQuat() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.quat.size(), robot_interface.quat.data());
    }

    py::array_t<float> GetRPY() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.rpy.size(), robot_interface.rpy.data());
    }


    py::array_t<float> GetProjectedGravity() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(
            robot_interface.projected_gravity.size(), 
            robot_interface.projected_gravity.data()
        );
    }

    py::array_t<float> lxy() {
        std::lock_guard<std::mutex> lock(state_mutex);
        std::array<float, 2> lxy = {gamepad.lx, gamepad.ly};
        return py::array_t<float>(lxy.size(), lxy.data());
    }

    py::array_t<float> rxy() {
        std::lock_guard<std::mutex> lock(state_mutex);
        std::array<float, 2> rxy = {gamepad.rx, gamepad.ry};
        return py::array_t<float>(rxy.size(), rxy.data());
    }

    py::array_t<float> GetKp() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.kp.size(), robot_interface.kp.data());
    }

    py::array_t<float> GetKd() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return py::array_t<float>(robot_interface.kd.size(), robot_interface.kd.data());
    }

    py::array_t<float> GetVelocity() {
        std::lock_guard<std::mutex> lock(state_mutex);
        auto velocity = high_state.velocity();
        return py::array_t<float>(velocity.size(), velocity.data());
    }

    py::array_t<float> GetFootPos() {
        std::lock_guard<std::mutex> lock(state_mutex);
        auto feet_pos = high_state.foot_position_body();
        return py::array_t<float>(feet_pos.size(), feet_pos.data());
    }

    py::array_t<float> GetFullState() {
        std::lock_guard<std::mutex> lock(state_mutex);
        std::array<float, 12 + 12 + 12 + 3 + 3 + 4> state;
        std::copy(
            robot_interface.jpos.begin(), 
            robot_interface.jpos.begin() + robot_interface.jpos.size(), 
            state.begin()
        );
        std::copy(
            robot_interface.jvel.begin(),
            robot_interface.jvel.begin() + robot_interface.jvel.size(),
            state.begin() + 12
        );
        std::copy(
            robot_interface.tau_est.begin(),
            robot_interface.tau_est.begin() + robot_interface.tau_est.size(),
            state.begin() + 24
        );
        std::copy(
            robot_interface.rpy.begin(),
            robot_interface.rpy.begin() + 3, 
            state.begin() + 36
        );
        std::copy(
            robot_interface.gyro.begin(),
            robot_interface.gyro.begin() + 3,
            state.begin() + 39
        );
        std::copy(
            low_state.foot_force().begin(),
            low_state.foot_force().begin() + 4,
            state.begin() + 42
        );
        return py::array_t<float>(state.size(), state.data());
    }

    py::float_ GetYawSpeed() {
        std::lock_guard<std::mutex> lock(state_mutex);
        auto yaw_speed = high_state.yaw_speed();
        return yaw_speed;
    }

    void SetKp(const float value) {
        robot_interface.kp.fill(value);
    }

    void SetKd(const float value) {
        robot_interface.kd.fill(value);
    }

    RobotState GetRobotState() {
        auto now = std::chrono::high_resolution_clock::now();
        robot_state.time_since_state_update = std::chrono::duration_cast<std::chrono::duration<double>>(now - timestamp_state).count();
        robot_state.time_since_control_update = std::chrono::duration_cast<std::chrono::duration<double>>(now - timestamp_command).count();
        robot_state.time_since_control_application = std::chrono::duration_cast<std::chrono::duration<double>>(now - timestamp_write).count();
        robot_state.control_application_since_state_update = std::chrono::duration_cast<std::chrono::duration<double>>(timestamp_write - timestamp_command).count();
        robot_state.jpos_des = robot_interface.jpos_des;
        robot_state.control_mode = control_mode;
        return robot_state;
    }

    RobotState robot_state;
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp_state, timestamp_command, timestamp_write;
    double interval_state, interval_command;
    bool lerp_command = false;  // whether to interpolate command between two control steps
    bool explicit_pd = false;   // whether to explicitly compute the torques from PD gains
    float damping_kd = 1.0;  // kd used for Damping mode

private:
    void LowStateMessageHandler(const void *message)
    {
        low_state = *(unitree_go::msg::dds_::LowState_ *)message;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            robot_interface.GetState(low_state);
            // update gamepad
            memcpy(rx.buff, &low_state.wireless_remote()[0], 40);
            gamepad.update(rx.RF_RX);
        }
        auto now = std::chrono::high_resolution_clock::now();
        robot_state.state_update_interval = std::chrono::duration_cast<std::chrono::duration<double>>(now - timestamp_state).count();
        timestamp_state = std::chrono::high_resolution_clock::now();
        
        robot_state.jpos = robot_interface.jpos;
        robot_state.jvel_raw = robot_interface.jvel;
        robot_state.tau_est = robot_interface.tau_est;
        
        auto imu = low_state.imu_state();
        robot_state.quat = imu.quaternion();
        robot_state.rpy = imu.rpy();
        robot_state.gyro = imu.gyroscope();
        robot_state.acc = imu.accelerometer();
        robot_state.foot_force = low_state.foot_force();
    }

    void HighStateMessageHandler(const void *message)
    {
        high_state = *(unitree_go::msg::dds_::SportModeState_ *)message;
    }

    void LowCmdwriteHandler()
    {
        // write low-level command
        {
            std::lock_guard<std::mutex> lock(cmd_mutex);
            lowcmd_publisher->Write(cmd);
        }
    }
    
    void MainThreadStep()
    {
        if (gamepad.L1.on_press && control_mode==0) {
            control_mode = 1;
            rsc.ServiceSwitch("sport_mode", 0, status);
            std::cout << "L1 pressed, switch mode to user control" << std::endl;
            UserControlCallback();
        } else if (gamepad.L1.on_press && control_mode==1) {
            control_mode = 0;
            rsc.ServiceSwitch("sport_mode", 1, status);
            std::cout << "L1 pressed, switch mode to sport mode" << std::endl;
        }

        if (substep % 2 == 0) {
            // jvel_substeps[(substep / 2) % 4] = robot_interface.jvel;
            // // average jvel over 4 substeps
            // for (int i = 0; i < 12; i++) {
            //     robot_state.jvel[i] = 0;
            //     for (int j = 0; j < 2; j++) {
            //         robot_state.jvel[i] += jvel_substeps[j][i];
            //     }
            //     robot_state.jvel[i] /= 4;
            // }
            lerp_(robot_state.jvel, robot_interface.jvel, 0.5);
        }
        substep +=1;

        // write low-level command
        for (int i = 0; i < 12; i++) {
            auto des = jpos_des[i];

            if (lerp_command) {
                auto now = std::chrono::high_resolution_clock::now();
                auto t = std::chrono::duration_cast<std::chrono::duration<double>>(now - timestamp_command).count();
                t = std::min(t / 0.02, 1.0);
                des = jpos_des_prev[i] + (jpos_des[i] - jpos_des_prev[i]) * t;
            }

            if (explicit_pd) {
                robot_interface.jpos_des[i] = robot_interface.jpos[i];
                robot_interface.jvel_des[i] = robot_interface.jvel[i];
                robot_interface.tau_ff[i] = robot_interface.kp[i] * (des - robot_interface.jpos[i]) + robot_interface.kd[i] * (0. - robot_interface.jvel[i]);
            } else {
                robot_interface.jpos_des[i] = des;
            }
        }
        
        if (control_mode == 1) {
            std::lock_guard<std::mutex> lock(cmd_mutex);

            robot_interface.SetCommand(cmd);
            lowcmd_publisher->Write(cmd);
            timestamp_write = std::chrono::high_resolution_clock::now();

            if (robot_interface.projected_gravity.at(2) > -0.1) {
                std::cout << "Falling detected, damping" << std::endl;
                Damping();
            }
        }
    }

    void Damping() {
        robot_interface.jvel_des.fill(0.);
        robot_interface.kp.fill(0.);
        robot_interface.kd.fill(damping_kd);
        robot_interface.tau_ff.fill(0.);
    }

    void UserControlCallback() {
        // robot_interface.jpos_des = {0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8};
        robot_interface.jvel_des.fill(0.);
        // robot_interface.kp.fill(30.);
        // robot_interface.kd.fill(1.0);
        robot_interface.tau_ff.fill(0.);
    }

protected:
    RobotInterface robot_interface;

    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> highstate_subscriber;

    ThreadPtr low_cmd_write_thread_ptr, control_thread_ptr, main_thread_ptr;

    unitree_go::msg::dds_::LowCmd_ cmd;
    unitree_go::msg::dds_::LowState_ low_state;
    unitree_go::msg::dds_::SportModeState_ high_state;

    // high-level
    go2::SportClient sport_client;
    go2::RobotStateClient rsc;
    std::array<float, 3> velocity;

    Gamepad gamepad;
    REMOTE_DATA_RX rx;

    std::mutex state_mutex, cmd_mutex;

    int control_mode=0; // 0: sportmode, 1: usermode
    int32_t status; // RobotStateClient service status

    std::array<float, 12> jpos_des, jpos_des_prev;
    std::array<std::array<float, 12>, 4> jvel_substeps;
    uint32_t substep = 0;
};

void InitChannel(const std::string &networkinterface) {
    unitree::robot::ChannelFactory::Instance()->Init(0, networkinterface);
}

PYBIND11_MODULE(go2py, m)
{
    py::class_<RobotState>(m, "RobotState")
        .def(py::init<>())
        .def_readonly("jpos", &RobotState::jpos)
        .def_readonly("jpos_des", &RobotState::jpos_des)
        .def_readonly("jvel", &RobotState::jvel)
        .def_readonly("jvel_raw", &RobotState::jvel_raw)
        .def_readonly("tau_est", &RobotState::tau_est)
        .def_readonly("quat", &RobotState::quat)
        .def_readonly("rpy", &RobotState::rpy)
        .def_readonly("gyro", &RobotState::gyro)
        .def_readonly("acc", &RobotState::acc)
        .def_readonly("foot_force", &RobotState::foot_force)
        .def_readonly("time_since_state_update", &RobotState::time_since_state_update)
        .def_readonly("time_since_control_update", &RobotState::time_since_control_update)
        .def_readonly("time_since_control_application", &RobotState::time_since_control_application)
        .def_readonly("control_application_since_state_update", &RobotState::control_application_since_state_update)
        .def_readonly("state_update_interval", &RobotState::state_update_interval)
        .def_readonly("control_mode", &RobotState::control_mode)
        ;

    py::class_<RobotIface>(m, "RobotIface")
        .def(py::init<>())
        .def("start_control", &RobotIface::StartControl, py::arg("interval") = 2000)
        .def("get_joint_pos", &RobotIface::GetJointPos)
        .def("get_joint_pos_target", &RobotIface::GetJointPosTarget)
        .def("get_joint_vel", &RobotIface::GetJointVel)
        .def("get_quat", &RobotIface::GetQuat)
        .def("get_rpy", &RobotIface::GetRPY)
        .def("get_projected_gravity", &RobotIface::GetProjectedGravity)
        .def("lxy", &RobotIface::lxy)
        .def("rxy", &RobotIface::rxy)
        .def("set_command", &RobotIface::SetCommand)
        .def("get_kd", &RobotIface::GetKd)
        .def("get_kp", &RobotIface::GetKp)
        .def("set_kd", &RobotIface::SetKd)
        .def("set_kp", &RobotIface::SetKp)
        .def("get_velocity", &RobotIface::GetVelocity)
        .def("get_feet_pos", &RobotIface::GetFootPos)
        .def("get_yaw_speed", &RobotIface::GetYawSpeed)
        .def("get_full_state", &RobotIface::GetFullState)
        .def("get_robot_state", &RobotIface::GetRobotState)
        .def_readwrite("lerp_command", &RobotIface::lerp_command)
        .def_readwrite("explicit_pd", &RobotIface::explicit_pd)
        .def_readwrite("damping_kd", &RobotIface::damping_kd)
        .def_readonly("timestamp_state", &RobotIface::timestamp_state)
        .def_readonly("timestamp_command", &RobotIface::timestamp_command)
        .def_readonly("interval_state", &RobotIface::interval_state)
        ;
    
    m.def("init_channel", &InitChannel, py::arg("networkinterface"));
}
