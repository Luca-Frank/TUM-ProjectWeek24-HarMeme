/*
 Navicat Premium Data Transfer

 Source Server         : wsl
 Source Server Type    : MySQL
 Source Server Version : 80031 (8.0.31)
 Source Host           : 172.23.171.25:3307
 Source Schema         : test_energy

 Target Server Type    : MySQL
 Target Server Version : 80031 (8.0.31)
 File Encoding         : 65001

 Date: 10/01/2024 09:13:26
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for model_performance
-- ----------------------------
DROP TABLE IF EXISTS `model_performance`;
CREATE TABLE `model_performance`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `model_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `binary_label` int NULL DEFAULT NULL,
  `precision` decimal(3, 2) NULL DEFAULT NULL,
  `recall` decimal(3, 2) NULL DEFAULT NULL,
  `f1_score` decimal(3, 2) NULL DEFAULT NULL,
  `support` int NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 11 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of model_performance
-- ----------------------------
INSERT INTO `model_performance` VALUES (1, 'AdaBoost', 0, 0.81, 0.84, 0.82, 230);
INSERT INTO `model_performance` VALUES (2, 'AdaBoost', 1, 0.68, 0.64, 0.66, 124);
INSERT INTO `model_performance` VALUES (3, 'Decision_Trees', 0, 0.82, 0.86, 0.84, 230);
INSERT INTO `model_performance` VALUES (4, 'Decision_Trees', 1, 0.71, 0.65, 0.68, 124);
INSERT INTO `model_performance` VALUES (5, 'Gradient_Boosting_Classifier', 0, 0.81, 0.88, 0.85, 230);
INSERT INTO `model_performance` VALUES (6, 'Gradient_Boosting_Classifier', 1, 0.74, 0.63, 0.68, 124);
INSERT INTO `model_performance` VALUES (7, 'BERT', 0, 0.85, 0.85, 0.85, 230);
INSERT INTO `model_performance` VALUES (8, 'BERT', 1, 0.72, 0.72, 0.72, 124);
INSERT INTO `model_performance` VALUES (9, 'Neural-Network', 0, 0.79, 0.75, 0.77, 230);
INSERT INTO `model_performance` VALUES (10, 'Neural-Network', 1, 0.58, 0.64, 0.61, 124);

SET FOREIGN_KEY_CHECKS = 1;
