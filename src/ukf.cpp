#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2.5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // first call check
  is_initialized_ = false;

  // set state dimension
  n_x_ = x_.size();

  // set augmented dimension
  n_aug_ = n_x_ + 2;

  // define spreading parameter
  lambda_ = 3.0 - n_aug_;

  // Number of Sigma Points
  n_sig_ = 2 * n_aug_ + 1;

  // Weights of sigma points
  weights_ = VectorXd(n_sig_);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  Xsig_pred_.fill(0.0);

  //initial weights 
  double weight0 = lambda_ / (lambda_ + n_aug_);
  double weight = 0.5/(lambda_ + n_aug_);
  weights_.fill(weight);
  weights_(0) = weight0;

  time_us_ = 0;

  P_aug_ = MatrixXd(n_aug_,n_aug_);

  // Lidar covarian
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_*std_laspx_, 0,
        0, std_laspy_*std_laspy_;

  // Radar covariance 
  R_radar_ = MatrixXd(3,3);
  R_radar_ <<  std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0,std_radrd_*std_radrd_; 

  // measurement matrix
  H_ = MatrixXd(2, 5);
  H_ << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;

  P_ = MatrixXd::Identity(n_x_, n_x_);
  P_(3,3) = 0.5;
  P_(4,4) = 0.5;

}

UKF::~UKF() {
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  // std::cout << "process measure ment start " << std::endl;

  if(!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rhodot = meas_package.raw_measurements_(2);
      double vx = rhodot * cos(phi);
      double vy = rhodot * sin(phi);
      // initialize state.
      x_ << rho * cos(phi), rho * sin(phi), sqrt(vx * vx + vy * vy), 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
      // initialize state.
      double lidar_px = meas_package.raw_measurements_(0);
      double lidar_py = meas_package.raw_measurements_(1);
      x_ << lidar_px, lidar_py, 0, 0, 0;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction(dt);

  if (MeasurementPackage::SensorType::LASER == meas_package.sensor_type_ && use_laser_) {
    UpdateLidar(meas_package);
  }
  else if (MeasurementPackage::SensorType::RADAR == meas_package.sensor_type_ && use_radar_) {
    UpdateRadar(meas_package);
  }
}

void UKF::Prediction(double dt) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */


  // generate sigma points 
  // std::cout << "prediction start" << std::endl;

  MatrixXd x_sig = MatrixXd(n_aug_, n_sig_);
  VectorXd x_aug = VectorXd(n_aug_);
  x_sig.fill(0.0);
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
  
  MatrixXd L = P_aug.llt().matrixL();

  double sqrtlmd = sqrt(lambda_ + n_aug_);
  MatrixXd sqrtlmdL = sqrtlmd * L;

  x_sig.col(0) = x_aug;
  for (int i = 0; i< n_aug_; ++i) {
    x_sig.col(i+1)        = x_aug + sqrtlmdL.col(i);
    x_sig.col(i+1+n_aug_) = x_aug - sqrtlmdL.col(i);
  }

  // predict each sigma point

  // std::cout << "predict each sigma point" << std::endl;

  Xsig_pred_.fill(0.0);

  for(int i=0; i<n_sig_; i++) {
    double px = x_sig(0, i);
    double py = x_sig(1, i);
    double v = x_sig(2 ,i);
    double yaw = x_sig(3, i);
    double yawd = x_sig(4, i);
    double nu_a = x_sig(5, i);
    double nu_yawdd= x_sig(6, i);

    double px_p, py_p;

    if (std::fabs(yawd) > 0.001) {
      px_p = px + v/yawd * (sin(yaw + yawd * dt) - sin(yaw));                        
      py_p = py + v /yawd * (cos(yaw) - cos(yaw+yawd*dt));                        
    } else {
      px_p = px + v*dt*cos(yaw);
      py_p = py + v*dt*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*dt;
    double yawd_p = yawd;

    // add noize 
    px_p = px_p + 0.5*nu_a*dt*dt*cos(yaw);
    py_p = py_p + 0.5*nu_a*dt*dt*sin(yaw);
    v_p  = v_p +  nu_a*dt;

    yaw_p = yaw_p + 0.5*nu_yawdd*dt*dt;
    yawd_p = yawd_p + nu_yawdd*dt;

    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;;
    Xsig_pred_(4, i) = yawd_p;
  }

  // predict mean and covariance
  x_.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {  
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  P_.fill(0.0);
  for (int i=0; i < n_sig_; ++i) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    if(x_diff(3) > M_PI) 
      x_diff(3) -= 2.*M_PI;
    
    if(x_diff(3) < -M_PI)
      x_diff(3) += 2.*M_PI;
      
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // std::cout << "update lidar start" << std::endl;
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  VectorXd z = meas_package.raw_measurements_;
  int n_z = z.size();
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_lidar_;
  MatrixXd Sinv = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Sinv;

  //estimate 
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  x_ = x_ + (K * y);

  if(x_(3) > M_PI) 
    x_(3) -= 2.*M_PI;
    
  if(x_(3) < -M_PI) 
     x_(3) += 2.*M_PI;

  P_ = (I - K * H_) * P_;

  double NIS_lidar = y.transpose() * Sinv * y;
  std::cout << "Lidar NIS : " << NIS_lidar << std::endl;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // std::cout << "update radar start" << std::endl;

  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  VectorXd z = meas_package.raw_measurements_;
  int n_z = z.size();

  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
  VectorXd z_pred = VectorXd(n_z);
  MatrixXd S = MatrixXd(n_z, n_z);

  for (int i = 0; i < n_sig_; ++i) {
      double px = Xsig_pred_(0, i);
      double py = Xsig_pred_(1, i);
      double v = Xsig_pred_(2, i);
      double yaw = Xsig_pred_(3, i);
      
      double v1 = cos(yaw)*v;
      double v2 = sin(yaw)*v;

      Zsig(0, i) = sqrt(px*px + py*py); ;
      Zsig(1, i) = atan2(py,px);  
      Zsig(2, i) = (px*v1 + py*v2) / sqrt(px*px + py*py);
  }

  z_pred.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);  
  }

  S.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    if(z_diff(1) > M_PI)
      z_diff(1) -= 2.*M_PI;
      
    if(z_diff(1) < -M_PI)
      z_diff(1) += 2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S = S + R_radar_;

  //UKF update
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  for (int i = 0; i < n_sig_; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;   

    if(z_diff(1) > M_PI) 
      z_diff(1) -= 2.*M_PI;
      
    if(z_diff(1) < -M_PI) 
      z_diff(1) += 2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    if (x_diff(3) > M_PI) 
       x_diff(3)-=2.*M_PI;
     
    if (x_diff(3) < -M_PI) 
       x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain 
  MatrixXd K = MatrixXd(n_x_, n_z); 
  MatrixXd Sinv = S.inverse();

  K = Tc * Sinv;

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  if (z_diff(1) > M_PI) 
     z_diff(1) -= 2.*M_PI;
   
  if (z_diff(1) < -M_PI) 
    z_diff(1) += 2.*M_PI;

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  MatrixXd Kt = K.transpose();
  P_ = P_ - K * S * Kt;

  double NIS_radar = z_diff.transpose() * Sinv  * z_diff;
  std::cout << "Radar NIS : " << NIS_radar << std::endl;
}