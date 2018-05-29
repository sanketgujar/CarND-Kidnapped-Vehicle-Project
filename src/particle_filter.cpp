/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 20;
	default_random_engine gen;

	double std_x     =std[0];
	double std_y     =std[1];
	double std_theta =std[2];

	normal_distribution<double> dist_x(x,std_x);
	normal_distribution<double> dist_y(y,std_y);
	normal_distribution<double> dist_theta(theta,std_theta);

	for (int i =0; i< num_particles ; i++){
		Particle current_particle;
		current_particle.id = i;
		current_particle.x = dist_x(gen);
		current_particle.y = dist_y(gen);
		current_particle.theta = dist_theta(gen);
		current_particle.weight = 1.0;
		weights.push_back(current_particle.weight);
		particles.push_back(current_particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	double std_x     =std_pos[0];
	double std_y     =std_pos[1];
	double std_theta =std_pos[2];



	for (int i =0 ; i < num_particles ; i++ ){
		double current_x = particles[i].x;
		double current_y = particles[i].y;
		double current_theta = particles[i].theta;

		double pred_x, pred_y, pred_theta;

		if (fabs(yaw_rate) < 0.0001){
			//the vehicle is not rotating
			pred_x = current_x + velocity*cos(current_theta)*delta_t;
			pred_y = current_y + velocity*sin(current_theta)*delta_t;
			pred_theta = current_theta;
		}
		else{
			pred_x = current_x + (velocity/yaw_rate)*(sin(current_theta + (yaw_rate*delta_t)) - sin(current_theta));
			pred_y = current_y + (velocity/yaw_rate)*(cos(current_theta)- cos(current_theta + (yaw_rate*delta_t)));
			pred_theta = current_theta + (yaw_rate*delta_t);
		}
		normal_distribution<double> dist_x(pred_x,std_x);
		normal_distribution<double> dist_y(pred_y,std_y);
		normal_distribution<double> dist_theta(pred_theta,std_theta);

		particles[i].x  = dist_x(gen);
		particles[i].y  = dist_y(gen);
		particles[i].theta  = dist_theta(gen);
	}
}



void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double sensor_range) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i =0 ; i < observations.size(); i++){
			double lowest_distance = sensor_range;
			int closest_landmark_id = -1;
			double observation_x = observations[i].x;
			double observation_y = observations[i].y;
			for (unsigned int j=0; j < predicted.size(); j++ ){
				//get the predicted and obs distance...
				double predicted_x = predicted[j].x;
				double predicted_y = predicted[j].y;
				int predicted_id  = predicted[j].id;
				double distance_pts  = dist(observation_x, observation_y, predicted_x, predicted_y);

				if (distance_pts <  lowest_distance){
					lowest_distance = distance_pts;
					closest_landmark_id = predicted_id;
				}
			}
			observations[i].id  = closest_landmark_id;  //associate with the closest landmark
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double weight_norm = 0.0;
	for (int i=0; i< num_particles; i++){
			double particle_x = particles[i].x;
			double particle_y = particles[i].y;
			double particle_theta = particles[i].theta;

			//transform vehicle coordinayte to map coordinate.
			vector<LandmarkObs> transformed_observations;
			for (unsigned int j = 0; j < observations.size(); j++){
					LandmarkObs transformed_obj;
					transformed_obj.x = particle_x + (cos(particle_theta)*observations[j].x) - (sin(particle_theta)*observations[j].y);
					transformed_obj.y = particle_y + (sin(particle_theta)*observations[j].x) + (cos(particle_theta)*observations[j].y);
					transformed_obj.id = j;
					transformed_observations.push_back(transformed_obj);
			}

			//filter map landmark outside the sensor range...
			vector<LandmarkObs> predicted_landmark;
			for (unsigned int j =0; j < map_landmarks.landmark_list.size(); j++){
					Map::single_landmark_s current_lan = map_landmarks.landmark_list[j];
					if ((fabs(particle_x - current_lan.x_f)<= sensor_range) && (fabs(particle_y - current_lan.y_f)<=sensor_range)){
						predicted_landmark.push_back(LandmarkObs {current_lan.id_i, current_lan.x_f, current_lan.y_f});
					}
			}

			//associate the precited and observation landmarks
			dataAssociation(predicted_landmark, transformed_observations, sensor_range);

			//calculating the weight of each particle..
			particles[i].weight= 1.0;
			double sigma_x_sq = pow(std_landmark[0],2);
			double sigma_y_sq = pow(std_landmark[1],2);
			double normalizer = (1. / 2.*M_PI*std_landmark[0]*std_landmark[1]);
			for (unsigned int k =0 ; k < transformed_observations.size(); k++){
					double trans_ob_x = transformed_observations[k].x;
					double trans_ob_y = transformed_observations[k].y;
					double trans_ob_id = transformed_observations[k].id;
					double multi_prob   = 1.0;

					for (unsigned int l =0; l < predicted_landmark.size(); l++){
							double predicted_landmark_x = predicted_landmark[l].x;
							double predicted_landmark_y = predicted_landmark[l].y;
							double predicted_landmark_id = predicted_landmark[l].id;

							if (trans_ob_id == predicted_landmark_id){
									multi_prob = normalizer*(exp(-1.*((pow((trans_ob_x -predicted_landmark_x),2)/(2.*sigma_x_sq)) +
								  							(pow((trans_ob_y - predicted_landmark_y),2)/(2.*sigma_y_sq)))));
									particles[i].weight *= multi_prob;
							}
					}
			}
			weight_norm += particles[i].weight;
	}

	//normalize the weights//
	for (unsigned int i = 0 ; i< particles.size(); i++){
		particles[i].weight /= weight_norm;
		weights[i]  = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> resampled_particles;
	default_random_engine gen;
	uniform_int_distribution<int> particle_index(0, num_particles-1);
	int current_index = particle_index(gen);

	double beta = 0.0;
	double max_weight_2 = 2.0* *max_element(weights.begin(), weights.end());

	for(unsigned int i =0; i < particles.size(); i++){
			uniform_real_distribution<double> random_weights(0., max_weight_2);
			beta += random_weights(gen);

			while(beta > weights[current_index]){
					beta -= weights[current_index];
					current_index = (current_index + 1) %num_particles;
			}
			resampled_particles.push_back(particles[current_index]);
	}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
		return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
