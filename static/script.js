import * as THREE from 'three';

import Stats from 'three/addons/libs/stats.module.js';
import { GPUStatsPanel } from 'three/addons/utils/GPUStatsPanel.js';

import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import * as GeometryUtils from 'three/addons/utils/GeometryUtils.js';

import {CSG} from "./threecsg.js"

var res = window.res;
var hc = window.hc;
var multi_high = window.multi_high;
var current_frame = 0;
var meshes = [];
var joint_meshes = [];
var rad = 0.03;

function make_linkage(x_length, y_length, z_length, hole_r) {
    hole_r = rad;
    var link = new THREE.BoxGeometry( x_length, y_length, z_length );
    var end_1 = new THREE.CylinderGeometry(y_length/2, y_length/2, z_length, 32);
    var end_2 = new THREE.CylinderGeometry(y_length/2, y_length/2, z_length, 32);
    var hole_1 = new THREE.CylinderGeometry(hole_r, hole_r, z_length, 32);
    var hole_2 = new THREE.CylinderGeometry(hole_r, hole_r, z_length, 32);
    end_1.rotateX(Math.PI/2).translate(-x_length/2, 0, 0);
    end_2.rotateX(Math.PI/2).translate(x_length/2, 0, 0);
    hole_1.rotateX(Math.PI/2).translate(-x_length/2, 0, 0);
    hole_2.rotateX(Math.PI/2).translate(x_length/2, 0, 0);

    var bsp_link = CSG.fromMesh(new THREE.Mesh(link));
    var hole_1 = CSG.fromMesh(new THREE.Mesh(hole_1));
    var hole_2 = CSG.fromMesh(new THREE.Mesh(hole_2));
    var bsp_end_1 = CSG.fromMesh(new THREE.Mesh(end_1)).subtract(bsp_link).subtract(hole_1);
    var bsp_end_2 = CSG.fromMesh(new THREE.Mesh(end_2)).subtract(bsp_link).subtract(hole_2);
    bsp_link = bsp_link.subtract(hole_1).subtract(hole_2);

    var full_link = CSG.toMesh(bsp_link.union(bsp_end_1).union(bsp_end_2));

    return full_link;
}

function make_link(x_length, y_length, z_length, hole_r, Theta_z, translation) {
    hole_r = rad;
    var full_link = make_linkage(x_length, y_length, z_length * 0.9, hole_r);
    full_link.rotation.z = Theta_z;
    full_link.position.set(translation[0] + x_length/2 * Math.cos(Theta_z), translation[1] + x_length/2 * Math.sin(Theta_z), translation[2]);
    return full_link;
}

function make_joint(x, y, z, height, radius=0.03) {
    radius = rad;
    var joint = new THREE.CylinderGeometry(radius, radius, height * 0.9, 32);
    joint.rotateX(Math.PI/2);
    joint = new THREE.Mesh(joint);
    joint.position.set(x, y, z);
    return joint;
}

(async function onLoad() {
    var res = window.res;
    var hc = window.hc;
    var multi_high = window.multi_high;
    // find the mean value of the res array
    var max_x = 0;
    var max_y = 0;
    var min_x = Infinity;
    var min_y = Infinity;
    for(let i = 0; i < res.length; i++) {
        for(let j = 0; j < res[i][1].length; j++) {
            if (res[i][1][j][0] > max_x) {
                max_x = res[i][1][j][0];
            }
            if (res[i][1][j][0] < min_x) {
                min_x = res[i][1][j][0];
            }
            if (res[i][1][j][1] > max_y) {
                max_y = res[i][1][j][1];
            }
            if (res[i][1][j][1] < min_y) {
                min_y = res[i][1][j][1];
            }
        }
    }
    var mean_x = (max_x + min_x) / 2;
    var mean_y = (max_y + min_y) / 2;

    // resize the res to a box from -1 to 1
    var scaling_factor = 8/Math.max(max_x - min_x, max_y - min_y);

    for (let i = 0; i < res.length; i++) {
        for (let j = 0; j < res[i][1].length; j++) {
            res[i][1][j][0] = (res[i][1][j][0] - mean_x) * scaling_factor;
            res[i][1][j][1] = (res[i][1][j][1] - mean_y) * scaling_factor;
        }
        for (let j = 0; j < res[i][0].length; j++) {
            res[i][0][j][0] = res[i][0][j][0] * scaling_factor;
            res[i][0][j][1] = res[i][0][j][1] * scaling_factor;
            res[i][0][j][2] = res[i][0][j][2] * scaling_factor;
            res[i][0][j][5][0] = (res[i][0][j][5][0] - mean_x) * scaling_factor;
            res[i][0][j][5][1] = (res[i][0][j][5][1] - mean_y) * scaling_factor;
        }
    }

    for (let i = 0; i < hc.length; i++) {
        hc[i][0] = (hc[i][0] - mean_x) * scaling_factor;
        hc[i][1] = (hc[i][1] - mean_y) * scaling_factor;
    }

    mean_x = 0;
    mean_y = 0;

    rad = res[0][0][0][2] * 0.7;

    var container, camera, scene, renderer, controls, last_frame_time;
    
    init();
    animate();
    controls.update();
    function init() {
        container = document.getElementById('container');
        
        renderer = new THREE.WebGLRenderer({
        antialias: true
        });
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        container.appendChild(renderer.domElement);

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0xfafafa);
        
        camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 10000);
        camera.position.set(mean_x,mean_y, 20);
        scene.add(camera);
        window.onresize = function() {
        renderer.setSize(window.innerWidth, window.innerHeight);
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        }
        
        var ambientLight = new THREE.AmbientLight(0xffffff, 2.0);
        scene.add(ambientLight);

        // var directionalLight = new THREE.DirectionalLight( 0xffffff, 0.5 );
        // directionalLight.position.x = 4;
        // directionalLight.position.y = 1;
        // directionalLight.position.z = -2;
        // scene.add( directionalLight );

      
        controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set(mean_x,mean_y,0);
        
        //   addGridHelper();
        createModel(0);
        last_frame_time = Date.now();
    }
    

    function createModel(j) {
        var results = res[j];
        //iterate through the results and create the links
        meshes = []
        for (let i = 0; i < results[0].length; i++) {
            var link = results[0][i];
            meshes.push(make_link(link[0], link[1], link[2], link[3], link[4], link[5]));
            if (i == 0){
                //yellow
                meshes[i].material = new THREE.MeshStandardMaterial({ color: 0xff8c00, opacity: 0.5, transparent: true , side: THREE.DoubleSide, depthWrite: true});
            }
            else
                meshes[i].material = new THREE.MeshStandardMaterial({ color: 0x3b3b3b, opacity: 0.5, transparent: true , side: THREE.DoubleSide, depthWrite: true});
            scene.add(meshes[i]);
        }
        joint_meshes = []
        for (let i = 0; i < results[1].length; i++) {
            var joint = results[1][i];
            joint_meshes.push(make_joint(joint[0], joint[1], joint[3], joint[2]));
            if(joint[4] == 1)//brown
                joint_meshes[i].material = new THREE.MeshStandardMaterial({ color: 0xff4c00, side: THREE.DoubleSide, depthWrite: false});
            else if(i<results[1].length-1)
                joint_meshes[i].material = new THREE.MeshStandardMaterial({ color: 0x121212, side: THREE.DoubleSide, depthWrite: false});
            else
                joint_meshes[i].material = new THREE.MeshStandardMaterial({ color: 0xe00043, side: THREE.DoubleSide, depthWrite: false});
            scene.add(joint_meshes[i]);
        }

        if (multi_high) {
            var n = hc.length;

            for (let j = 0; j < n; j++) {
                var points = [];
                var colors = [];
                for (let i = 0; i <hc[j].length; i++) {
                    points.push(hc[j][i][0], hc[j][i][1], hc[j][i][2]);
                }
                
                var geometry = new LineGeometry();
                geometry.setPositions( points );
                var material = new LineMaterial( {
                    color: 0xff4c00,
                    linewidth: 0.002, // in world units with size attenuation, pixels otherwise
                    vertexColors: false,

                    //resolution:  // to be set by renderer, eventually
                    dashed: false,
                    alphaToCoverage: false,
                    depthWrite: false,
                    depthTest: false,
                    transparent: true,

                });
                material.worldUnits = false;
                const line = new Line2(geometry, material);
                line.computeLineDistances();
                line.scale.set( 1, 1, 1 );
                line.renderOrder= 9999;
                scene.add(line);
            }
        }
        else{
            var points = [];
            var colors = [];
            for (let i = 0; i <hc.length; i++) {
                points.push(hc[i][0], hc[i][1], hc[i][2]);
            }
            
            var geometry = new LineGeometry();
            geometry.setPositions( points );
            var material = new LineMaterial( {
                color: 0xe00043,
                linewidth: 0.002, // in world units with size attenuation, pixels otherwise
                vertexColors: false,

                //resolution:  // to be set by renderer, eventually
                dashed: false,
                alphaToCoverage: false,
                depthWrite: false,
                depthTest: false,
                transparent: true,

            });
            material.worldUnits = false;
            const line = new Line2(geometry, material);
            line.computeLineDistances();
            line.scale.set( 1, 1, 1 );
            line.renderOrder= 9999;
            scene.add(line);
        }

        
    }
  
    function addGridHelper() {
  
      var helper = new THREE.GridHelper(10, 10);
      helper.material.opacity = 0.25;
      helper.material.transparent = true;
      scene.add(helper);
  
      var axis = new THREE.AxesHelper(100);
      scene.add(axis);
    }

    function set_idx(){
        if(current_frame < res.length-1) {
            // createModel(current_frame);
            current_frame += 1;
        }
        else {
            current_frame = 0;
        }
    }
  
    function animate() {
        
        if (Date.now() - last_frame_time > 1000/30) {
            set_idx();
            last_frame_time = Date.now();
        }

        for (let i = 0; i < meshes.length; i++) {
            meshes[i].rotation.z = res[current_frame][0][i][4];
            meshes[i].position.set(res[current_frame][0][i][5][0] + res[current_frame][0][i][0]/2 * Math.cos(res[current_frame][0][i][4]), res[current_frame][0][i][5][1] + res[current_frame][0][i][0]/2 * Math.sin(res[current_frame][0][i][4]), res[current_frame][0][i][5][2]);
        }

        for (let i = 0; i < joint_meshes.length; i++) {
            joint_meshes[i].position.set(res[current_frame][1][i][0], res[current_frame][1][i][1], res[current_frame][1][i][3]);
        }
         
        requestAnimationFrame(animate);
        render();
    }
  
    function render() {
      renderer.clearDepth();
      renderer.render(scene, camera);
    }
  })();