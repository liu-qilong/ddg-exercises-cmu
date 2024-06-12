"use strict";

/**
 * @module Projects
 */
class SimplicialComplexOperators {

        /** This class implements various operators (e.g. boundary, star, link) on a mesh.
         * @constructor module:Projects.SimplicialComplexOperators
         * @param {module:Core.Mesh} mesh The input mesh this class acts on.
         * @property {module:Core.Mesh} mesh The input mesh this class acts on.
         * @property {module:LinearAlgebra.SparseMatrix} A0 The vertex-edge adjacency matrix of <code>mesh</code>.
         * @property {module:LinearAlgebra.SparseMatrix} A1 The edge-face adjacency matrix of <code>mesh</code>.
         */
        constructor(mesh) {
                this.mesh = mesh;
                this.assignElementIndices(this.mesh);

                this.A0 = this.buildVertexEdgeAdjacencyMatrix(this.mesh);
                this.A1 = this.buildEdgeFaceAdjacencyMatrix(this.mesh);
        }

        /** Assigns indices to the input mesh's vertices, edges, and faces
         * @method module:Projects.SimplicialComplexOperators#assignElementIndices
         * @param {module:Core.Mesh} mesh The input mesh which we index.
         */
        assignElementIndices(mesh) {
                mesh.vertices.forEach((vertex, idx) => vertex.index = idx)
                mesh.edges.forEach((edge, idx) => edge.index = idx)
                mesh.faces.forEach((face, idx) => face.index = idx)
                return mesh
        }

        /** Returns the vertex-edge adjacency matrix of the given mesh.
         * @method module:Projects.SimplicialComplexOperators#buildVertexEdgeAdjacencyMatrix
         * @param {module:Core.Mesh} mesh The mesh whose adjacency matrix we compute.
         * @returns {module:LinearAlgebra.SparseMatrix} The vertex-edge adjacency matrix of the given mesh.
         */
        buildVertexEdgeAdjacencyMatrix(mesh) {
                let T = new Triplet(mesh.edges.length, mesh.vertices.length)
                
                for (let edge of mesh.edges) {
                        T.addEntry(1, edge.index, edge.halfedge.vertex.index)
                        T.addEntry(1, edge.index, edge.halfedge.twin.vertex.index)
                }

                return SparseMatrix.fromTriplet(T)
        }

        /** Returns the edge-face adjacency matrix.
         * @method module:Projects.SimplicialComplexOperators#buildEdgeFaceAdjacencyMatrix
         * @param {module:Core.Mesh} mesh The mesh whose adjacency matrix we compute.
         * @returns {module:LinearAlgebra.SparseMatrix} The edge-face adjacency matrix of the given mesh.
         */
        buildEdgeFaceAdjacencyMatrix(mesh) {
                let T = new Triplet(mesh.faces.length, mesh.edges.length)

                for (let face of mesh.faces) {
                        for (let halfedge of face.adjacentHalfedges()) {
                                T.addEntry(1, face.index, halfedge.edge.index)
                        }
                }

                return SparseMatrix.fromTriplet(T)
        }

        /** Returns a column vector representing the vertices of the
         * given subset.
         * @method module:Projects.SimplicialComplexOperators#buildVertexVector
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |V| entries. The ith entry is 1 if
         *  vertex i is in the given subset and 0 otherwise
         */
        buildVertexVector(subset) {
                let T = new Triplet(this.mesh.vertices.length, 1)
                
                for (let vertex of subset.vertices) {
                        T.addEntry(1, vertex, 0)
                }

                return SparseMatrix.fromTriplet(T)
        }

        /** Returns a column vector representing the edges of the
         * given subset.
         * @method module:Projects.SimplicialComplexOperators#buildEdgeVector
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |E| entries. The ith entry is 1 if
         *  edge i is in the given subset and 0 otherwise
         */
        buildEdgeVector(subset) {
                let T = new Triplet(this.mesh.edges.length, 1)
                
                for (let edge of subset.edges) {
                        T.addEntry(1, edge, 0)
                }

                return SparseMatrix.fromTriplet(T)
        }

        /** Returns a column vector representing the faces of the
         * given subset.
         * @method module:Projects.SimplicialComplexOperators#buildFaceVector
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:LinearAlgebra.DenseMatrix} A column vector with |F| entries. The ith entry is 1 if
         *  face i is in the given subset and 0 otherwise
         */
        buildFaceVector(subset) {
                let T = new Triplet(this.mesh.faces.length, 1)
                
                for (let face of subset.faces) {
                        T.addEntry(1, face, 0)
                }

                return SparseMatrix.fromTriplet(T)
        }

        /** Returns the star of a subset.
         * @method module:Projects.SimplicialComplexOperators#star
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:Core.MeshSubset} The star of the given subset.
         */
        star(subset) {
                let star_set = MeshSubset.deepCopy(subset)

                // add edges that contains vertices in the subset
                let vertex_vec = this.buildVertexVector(star_set)
                let star_edge_vec = this.A0.timesSparse(vertex_vec).toDense()

                for (let i = 0; i < star_edge_vec.nRows(); i++) {
                        if (star_edge_vec.get(i, 0) > 0) {
                                star_set.addEdge(i)
                        }
                }

                // add faces that contains edges in the subset
                let edge_vec = this.buildEdgeVector(star_set)
                let star_face_vec = this.A1.timesSparse(edge_vec).toDense()

                for (let i = 0; i < star_face_vec.nRows(); i++) {
                        if (star_face_vec.get(i, 0) > 0) {
                                star_set.addFace(i)
                        }
                }

                return star_set
        }

        /** Returns the closure of a subset.
         * @method module:Projects.SimplicialComplexOperators#closure
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:Core.MeshSubset} The closure of the given subset.
         */
        closure(subset) {                
                let closure_set = MeshSubset.deepCopy(subset)

                // add edges of all faces in the star
                let face_vec = this.buildFaceVector(closure_set)
                let closure_edge_vec = face_vec.transpose().timesSparse(this.A1).toDense()

                for (let i = 0; i < closure_edge_vec.nCols(); i++) {
                        if (closure_edge_vec.get(0, i) > 0) {
                                closure_set.addEdge(i)
                        }
                }

                // add vertices of all edges in the closure
                let edge_vec = this.buildEdgeVector(closure_set)
                let closure_vertex_vec = edge_vec.transpose().timesSparse(this.A0).toDense()

                for (let i = 0; i < closure_vertex_vec.nCols(); i++) {
                        if (closure_vertex_vec.get(0, i) > 0) {
                                closure_set.addVertex(i)
                        }
                }

                return closure_set
        }

        /** Returns the link of a subset.
         * @method module:Projects.SimplicialComplexOperators#link
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {module:Core.MeshSubset} The link of the given subset.
         */
        link(subset) {
                // seems wrong!!!
                let star_set = this.star(subset)
                let link_set = this.closure(subset)
                link_set.deleteSubset(star_set)
                return link_set
        }

        /** Returns true if the given subset is a subcomplex and false otherwise.
         * @method module:Projects.SimplicialComplexOperators#isComplex
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {boolean} True if the given subset is a subcomplex and false otherwise.
         */
        isComplex(subset) {
                return subset.equals(this.closure(subset))
        }

        /** Returns the degree if the given subset is a pure subcomplex and -1 otherwise.
         * @method module:Projects.SimplicialComplexOperators#isPureComplex
         * @param {module:Core.MeshSubset} subset A subset of our mesh.
         * @returns {number} The degree of the given subset if it is a pure subcomplex and -1 otherwise.
         */
        isPureComplex(subset) {
                if (subset.equals(new MeshSubset())) {
                        // empty set
                        return -1
                } else if (!this.isComplex(subset)) {
                        // not a subcomplex
                        return -1
                } else {
                        if (subset.edges.size === 0) {
                                // only vertices
                                return 0
                        } else {
                                // have edges
                                let vertex_vec = this.buildVertexVector(subset).transpose().toDense()
                                let edge_vec = this.buildEdgeVector(subset).transpose()
                                let degrees = edge_vec.timesSparse(this.A0).toDense()
                                let degree = degrees.get(0, 0)

                                for (let i = 0; i < degrees.nCols(); i++) {
                                        if (vertex_vec.get(0, i) == 1 && degrees.get(0, i) !== degree) {
                                                return -1
                                        }
                                }

                                return degree
                        }
                }
        }

        /** Returns the boundary of a subset.
         * @method module:Projects.SimplicialComplexOperators#boundary
         * @param {module:Core.MeshSubset} subset A subset of our mesh. We assume <code>subset</code> is a pure subcomplex.
         * @returns {module:Core.MeshSubset} The boundary of the given pure subcomplex.
         */
        boundary(subset) {
                if (this.isPureComplex(subset) === -1) {
                        console.log('Not a pure subcomplex')
                        return subset
                } else {
                        let boundary_set = new MeshSubset()

                        // add vertices that belongs to only 1 edge
                        let edge_vec = this.buildEdgeVector(subset).transpose()
                        let degrees = edge_vec.timesSparse(this.A0).toDense()

                        for (let i = 0; i < degrees.nCols(); i++) {
                                if (degrees.get(0, i) == 1) {
                                        boundary_set.addVertex(i)
                                }
                        }

                        // add edges that belongs to only 1 face
                        let face_vec = this.buildFaceVector(subset).transpose()
                        degrees = face_vec.timesSparse(this.A1).toDense()

                        for (let i = 0; i < degrees.nCols(); i++) {
                                if (degrees.get(0, i) == 1) {
                                        boundary_set.addEdge(i)
                                }
                        }

                        return this.closure(boundary_set)
                }
        }
}