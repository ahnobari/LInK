import * as THREE from 'three';

import Stats from 'three/addons/libs/stats.module.js';
import { GPUStatsPanel } from 'three/addons/utils/GPUStatsPanel.js';

import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import * as GeometryUtils from 'three/addons/utils/GeometryUtils.js';