# Plasmo Interface to Ipopt
# Yankai Cao, Victor Zavala
# UW-Madison, 2016

using MathProgBase.SolverInterface
using Ipopt, JuMP

mutable struct IpoptModelData
    d
    relative_n::Int
    n::Int
    m::Int
    jacnnz::Int
    hessnnz::Int
    numconnectRows::Int
end
IpoptModelData() = IpoptModelData(nothing,0,0,0,0,0,0)

#=
function getData(m::JuMP.Model)
    if haskey(m.ext, :Data)
        return m.ext[:Data]
    else
        error("This functionality is only available")
    end
end
=#

function setLeafModelList(master::JuMP.Model)
	 numLeaf = 0
         children = getchildren(master)
	 modelList = JuMP.Model[]
         if children == []
	      modelList = [modelList; master]
	      numLeaf = 1
              return numLeaf, modelList
         end
         for (idx,child) in enumerate(children)
	     child_numLeaf, child_modelList = setLeafModelList(child)
	     modelList = [modelList; child_modelList]
	     numLeaf += child_numLeaf
         end
         return numLeaf, modelList
end

function setUpModelList(master::JuMP.Model)
         children = getchildren(master)
         modelList = JuMP.Model[]
         if children == []
              return modelList
         end
         for (idx,child) in enumerate(children)
             child_modelList = setUpModelList(child)
             modelList = [modelList; child_modelList]
         end
	 modelList = [modelList; master]
         return modelList 
end


function Ipopt_solve(master::JuMP.Model)
	 numLeaf , leafModelList = setLeafModelList(master)
         upModelList = setUpModelList(master)
         modelList = [leafModelList; upModelList]
	 net_rowLb = Float64[]
         net_rowUb = Float64[]
         net_colLb = Float64[]
         net_colUb = Float64[]
         net_initval = Float64[]
         local_initval = Float64[]
         local_lowbound = Float64[]
         local_upbound = Float64[]
         net_m = 0
         net_n = 0
         net_hessnnz = 0
         net_jacnnz = 0
	 
         for (idx,node) in enumerate(modelList)
	     node.ext[:Data] = IpoptModelData()
             local_data = getData(node)
             # Bounds are constants/coefficients of constraints
             #nlp_lb, nlp_ub = JuMP.constraintbounds(node)
             #net_rowLb = [net_rowLb;nlp_lb]
             #net_rowUb = [net_rowUb;nlp_ub]
             for var in all_variables(node)
                try
                    push!(local_initval,JuMP.start_value(var))
                    push!(local_lowbound,JuMP.lower_bound(var))
                    push!(local_upbound,JuMP.upper_bound(var))
                catch e
                    continue
                end
             end
             if any(isnan,local_initval)
                for (indx,colm) in enumerate(local_initval)
                    if isnan(colm)
                        local_initval[indx] = 0
                    end
                end
                #local_initval[isnan(node.colVal)] .= 0
                
                local_initval = min(max(local_lowbound,local_initval),local_upbound)
             end
             net_initval = [net_initval;local_initval]
             #local_data.m = length(nlp_lb)
             local_data.n = num_variables(node)
             local_data.relative_n = net_n
	     #net_m += local_data.m
             net_n += local_data.n
         end

	 Icon = Int[]
         Jcon = Int[]
         Vcon = Float64[]
         numconnectRows = 0

	 deleted = []
	 for (idx,node) in enumerate(upModelList)
	     connectRows,contypes = getConnectRows(node)
	     s = 1	
             for c in connectRows
             	 coeffs = values(JuMP.constraint_object(c).func.terms)
             	 vars   = keys(JuMP.constraint_object(c).func.terms)
             	 for (it,ind) in enumerate(coeffs)
                     push!(Icon, numconnectRows + s)
                     push!(Jcon, getData(vars[it].m).relative_n + vars[it].col)
                     push!(Vcon, ind)
             	 end
		 s = s + 1
             end
	     numconnectRows += length(connectRows)
	     local_data = getData(node)
             local_data.numconnectRows = length(connectRows)
	     local_data.m = local_data.m - local_data.numconnectRows	  
         if length(connectRows)>=1
            push!(deleted, connectRows)
            for row in connectRows
                delete(node,row)
            end
        end
	 end
	 Mcon = sparse(Icon, Jcon, Vcon, numconnectRows, net_n)

	 for (idx,node) in enumerate(modelList)
	    local_data = getData(node) 
            NLPeval = JuMP.NLPEvaluator(node)
            MOI.initialize(NLPeval, [:Grad,:Jac,:Hess])
            Ihess, Jhess = hesslag_structure(local_data.d)
            local_data.hessnnz = length(Ihess)
            net_hessnnz += local_data.hessnnz

            Ijac, Jjac = jac_structure(local_data.d)
            local_data.jacnnz = length(Ijac)
            net_jacnnz += local_data.jacnnz
	 end
    	 net_jacnnz += length(Icon)


         function eval_f_cb(x)
                f = 0
                n_start = 1
                n_end = 0
                for (idx,node) in enumerate(modelList)
                    local_data = getData(node)
                    n_end += local_data.n
                    local_d = getData(node).d
                    local_x = x[n_start:n_end]
                    local_scl = (node.objSense == :Min) ? 1.0 : -1.0
                    f += local_scl*eval_f(local_d,local_x)
                    n_start += local_data.n
                end
                return f
         end

         function eval_g_cb(x, g)
                n_start = 1
                m_start = 1
                n_end = 0
                m_end = 0
                for (idx,node) in enumerate(modelList)
                    local_data = getData(node)
                    n_end += local_data.n
                    m_end += local_data.m
                    local_d = getData(node).d
                    local_x = x[n_start:n_end]
                    local_g = g[m_start:m_end]
                    eval_g(local_d, local_g, local_x)
                    g[m_start:m_end] = local_g
                    n_start += local_data.n
                    m_start += local_data.m
                end
                g[m_start:m_start+numconnectRows-1] = Mcon * x
		return Int32(1)
         end

         function eval_grad_f_cb(x, grad_f)
                n_start = 1
                n_end = 0
                for (idx,node) in enumerate(modelList)
                    local_data = getData(node)
                    n_end += local_data.n
                    local_d = getData(node).d
                    local_x = x[n_start:n_end]
                    local_grad_f = grad_f[n_start:n_end]
                    eval_grad_f(local_d, local_grad_f, local_x)
                    local_scl = (node.objSense == :Min) ? 1.0 : -1.0
                    scale!(local_grad_f,local_scl)
                    grad_f[n_start:n_end] = local_grad_f
                    n_start += local_data.n
                end
                return Int32(1)
         end

         function eval_jac_g_cb(x, mode, rows, cols, values)
               n_start = 1
               m_start = 1
               nnz_start = 1
               n_end = 0
               nnz_end = 0
               if mode == :Structure
                  for (idx,node) in enumerate(modelList)
                      local_data = getData(node)
                      Ijac, Jjac = jac_structure(local_data.d)
                      for i in 1:length(Ijac)
                          rows[i+nnz_start-1] = Ijac[i] + m_start - 1
                          cols[i+nnz_start-1] = Jjac[i] + n_start - 1
                      end
                      n_start += local_data.n
                      nnz_start += length(Ijac)
                      m_start += local_data.m
                  end
                  rows[nnz_start:nnz_start+length(Icon)-1]  = Icon + net_m - numconnectRows
                  cols[nnz_start:nnz_start+length(Icon)-1]  = Jcon
               else
                  for (idx,node) in enumerate(modelList)
                      local_data = getData(node)
                      n_end += local_data.n
                      nnz_end += local_data.jacnnz
                      local_x = x[n_start:n_end]
                      local_values = values[nnz_start:nnz_end]
                      eval_jac_g(local_data.d, local_values, local_x)
                      values[nnz_start:nnz_end] = local_values
                      n_start += local_data.n
                      nnz_start += local_data.jacnnz
                  end
                  values[nnz_start:nnz_start+length(Icon)-1]  = Vcon
               end
               return Int32(1)
         end

         function eval_h_cb(x, mode, rows, cols, obj_factor,lambda, values)
            n_start = 1
            m_start = 1
            nnz_start = 1
            n_end = 0
            m_end = 0
            nnz_end = 0
            if mode == :Structure
               for (idx,node) in enumerate(modelList)
                   local_data = getData(node)
                   Ihess, Jhess = hesslag_structure(local_data.d)
                   for i in 1:length(Ihess)
                       rows[i+nnz_start-1] = Ihess[i] + n_start - 1
                       cols[i+nnz_start-1] = Jhess[i] + n_start - 1
                   end
                   n_start += local_data.n
                   nnz_start += length(Ihess)
               end
            else
               for (idx,node) in enumerate(modelList)
                   local_data = getData(node)
                   n_end += local_data.n
                   m_end += local_data.m
                   nnz_end += local_data.hessnnz
                   local_x = x[n_start:n_end]
                   local_lambda = lambda[m_start:m_end]
                   local_values = values[nnz_start: nnz_end]
                   local_scl = (node.objSense == :Min) ? 1.0 : -1.0
                   eval_hesslag(local_data.d, local_values, local_x, obj_factor*local_scl, local_lambda)
                   values[nnz_start:nnz_end]  = local_values
                   n_start += local_data.n
                   m_start += local_data.m
                   nnz_start += local_data.hessnnz
               end
            end
            return Int32(1)
         end


         prob = createProblem(net_n, float(net_colLb), float(net_colUb), net_m,
             float(net_rowLb), float(net_rowUb), net_jacnnz, net_hessnnz,
             eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb,
             eval_h_cb)

         prob.x = net_initval
         #addOption(prob, "max_iter", 2)
	 addOption(prob, "linear_solver", "ma57")
         status = solveProblem(prob)

         #println(prob.x)
	 n_start = 1
	 n_end = 0
	 for (idx,node) in enumerate(modelList)
	     local_data = getData(node)
             n_end += local_data.n
	     node.colVal = prob.x[n_start:n_end]

	     local_d = getData(node).d
	     node.objVal = eval_f(local_d, node.colVal)
	     #println("idx:    ", idx, "   objVal:  ", node.objVal)
	     n_start += local_data.n
	 end

         for (idx,node) in enumerate(upModelList)
	     node.linconstr = [node.linconstr; deleted[idx]]
         end

	 if status == 0
	    return :Optimal
	 else
	    return :NotOptimal   
	 end   
         #return Int32(1)
end



function getConnectRows(master::JuMP.Model)
    connectRows = []
    
    greaterThan = [all_constraints(master,GenericAffExpr{Float64,VariableRef},MOI.GreaterThan{Float64})]
    equalTo = [all_constraints(master,GenericAffExpr{Float64,VariableRef},MOI.EqualTo{Float64})]
    lessThan = [all_constraints(master,GenericAffExpr{Float64,VariableRef},MOI.LessThan{Float64})]
    alltypes = [greaterThan;equalTo;lessThan]
    
    for type in alltypes
        for constraint in type
                conobject = JuMP.constraint_object(constraint)
                coeffs = [values(conobject.func.terms)]
                vars   = [keys(conobject.func.terms)]
                for (i,coeff) in enumerate(coeffs)
                    try check_belongs_to_model(vars[i],master)
                       continue
                    catch e
                        push!(connectRows, constraint)
                    end
                end
        end
    end
    return connectRows,alltypes
end
