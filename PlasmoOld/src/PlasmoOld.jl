# Plasmo Code
# Yankai Cao, Victor Zavala
# UW-Madison, 2016
# objective cannot be nonlinear    (can be quadratic)

module PlasmoOld
using JuMP
import DataStructures.OrderedDict
export NetModel, @addNode, getNode, @Linkingconstraint
export getparent, getchildren, getchildrenDict
export Ipopt_solve
export PipsNlp_solve
export getsumobjectivevalue, RandomStochasticModel, StochasticModel, copyStoModel, copyModel, copyNLModel, extensiveSimplifiedModel, addNLconstraint2, getData, _splicevars!, copyNet, NetModel
using Base.Meta
using MathProgBase
using Distributions

#Check for nonlinear objective
function _has_nonlinear_obj(m::JuMP.Model)
    if m.nlp_data != nothing
        if m.nlp_data.nlobj != nothing
            return true
        end
    end
    return false
end

# COPY CONSTRAINT FUNCTIONS
function _copy_constraint_func(func::JuMP.GenericAffExpr,ref_map::Dict{VariableRef,VariableRef})
    terms = func.terms
    new_terms = OrderedDict([(ref_map[var_ref],coeff) for (var_ref,coeff) in terms])
    new_func = JuMP.GenericAffExpr{Float64,JuMP.VariableRef}()
    new_func.terms = new_terms
    new_func.constant = func.constant
    return new_func
end

function _copy_constraint_func(func::JuMP.GenericQuadExpr,ref_map::Dict{VariableRef,JuMP.VariableRef})
    new_aff = _copy_constraint_func(func.aff,ref_map)
    new_terms = OrderedDict([(JuMP.UnorderedPair(ref_map[pair.a],ref_map[pair.b]),coeff) for (pair,coeff) in func.terms])
    new_func = JuMP.GenericQuadExpr{Float64,JuMP.VariableRef}()
    new_func.terms = new_terms
    new_func.aff = new_aff
    #new_func.constant = func.constant
    return new_func
end

function _copy_constraint_func(func::JuMP.VariableRef,ref_map::Dict{VariableRef,VariableRef})
    new_func = ref_map[func]
    return new_func
end

function _copy_constraint(constraint::JuMP.ScalarConstraint,m::Model)
    ref_map = m.ext[:v_map]
    new_func = _copy_constraint_func(constraint.func,ref_map)
    new_con = JuMP.ScalarConstraint(new_func,constraint.set)
    return new_con
end

function _copy_constraint(constraint::JuMP.VectorConstraint,m::Model)
    new_funcs = [_copy_constraint_func(func,m.ext[:v_map]) for func in constraint.func]
    new_con = JuMP.VectorConstraint(new_funcs,constraint.set,constraint.shape)
    return new_con
end

function copy_extension_data(data::Array{Any,1},new_model::Model,model::Model)
    
end

#COPY OBJECTIVE FUNCTIONS
function _copy_objective(m::JuMP.Model,ref_map::Dict{VariableRef,VariableRef})
    return _copy_objective(JuMP.objective_function(m),ref_map)
end

function _copy_objective(func::Union{JuMP.GenericAffExpr,JuMP.GenericQuadExpr},ref_map::Dict{VariableRef,VariableRef})
    new_func = _copy_constraint_func(func,ref_map)
    return new_func
end

function _copy_objective(func::JuMP.VariableRef,ref_map::Dict{VariableRef,VariableRef})
    new_func = ref_map[func]
    return new_func
end

mutable struct NetData
    children::Vector{JuMP.Model}
    parent
    childrenDict::Dict{String, JuMP.Model}
end
NetData() = NetData(JuMP.Model[], nothing, Dict{String, JuMP.Model}())
#NetData() = NetData(nothing, Dict{String, JuMP.Model}())

function NetModel(buildType="serial")
    m = JuMP.Model()
    m.ext[:Net] = NetData()
    m.ext[:LinkDict] = Dict{JuMP.VariableRef, JuMP.VariableRef}()
    m.ext[:BuildType] = buildType
    m.ext[:linkingId] = []
    return m
end

# Copies over a NetModel and makes a reference map.
# Variable indices don't matter as first stage variables
# are recognized by name, not by index.

function copyNet(model::Model)
    caching_mode = backend(model).mode
    new_model = Model(caching_mode = caching_mode)

    # Copy the MOI backend, note that variable and constraint indices may have
    # changed, the `index_map` gives the map between the indices of
    # `backend(model` and the indices of `backend(new_model)`.
    index_map = MOI.copy_to(backend(new_model), backend(model),copy_names = true)
    
    # Hook into a solve call...function of the form f(m::Model; kwargs...),
    # where kwargs get passed along to subsequent solve calls.
    new_model.optimize_hook = model.optimize_hook

    # Copy NLP data
    if model.nlp_data !== nothing
        error("copy is not supported yet for models with nonlinear constraints and/or nonlinear objective function")
    end

    # Generate a reference map between indices of old and new variables
    reference_map = ReferenceMap(new_model, index_map)

    # Copy object dictionary over to new model
    for (name, value) in object_dictionary(model)
        new_model.obj_dict[name] = getindex.(reference_map, value)
    end
    
    # Copy extension data over to new model
    for (key, data) in model.ext
        new_model.ext[key] = model.ext[key]
    end

    return new_model
end


function getNet(m::JuMP.Model)
    if haskey(m.ext, :Net)
        return m.ext[:Net]
    else
        error("This functionality is only available to NetModel")
    end
end


function getData(m::JuMP.Model)
    if haskey(m.ext, :Data)
        return m.ext[:Data]
    else
        error("This functionality is only available")
    end
end

getparent(m::JuMP.Model)       = getNet(m).parent
getchildrenDict(m::JuMP.Model)     = getNet(m).childrenDict

getname(c::Symbol) = c
getname(c::Nothing) = ()
getname(c::Expr) = c.args[1]

function getchildren(m::JuMP.Model)
    if haskey(m.ext, :Net)
        return getNet(m).children
    else
        return []
    end
end

function getNode(m::JuMP.Model, modelname::String)
    if !haskey(getNet(m).childrenDict, modelname)
        error("No model with name $modelname")
    elseif getNet(m).childrenDict[modelname] === nothing
        error("Multiple models with name $modelname")
    else
        return getNet(m).childrenDict[modelname]
    end
end


function registermodel(m::JuMP.Model, modelname::String, value::JuMP.Model)
    if haskey(getNet(m).childrenDict, modelname)
        getNet(m).childrenDict[modelname] = nothing # indicate duplicate
        error("Multiple models with name $modelname")
    else
        getNet(m).childrenDict[modelname] = value
    end
end

macro Linkingconstraint(m, args...)
      expr = quote
            id_start = length($(m).linconstr) + 1
          @constraint($(m),$(args...))
      id_end = length($(m).linconstr)
      $(m).ext[:linkingId] = [$(m).ext[:linkingId]; id_start:id_end]
      end
      return esc(expr)
end

macro addNode(m, node)
    if isa(node, Symbol)
       return quote
           if haskey($(esc(node)).ext, :Net)
            getNet($(esc(node))).parent = $(esc(m))
           else
        $(esc(node)).ext[:Net] = NetData(JuMP.Model[], $(esc(m)),Dict{Symbol, JuMP.Model}())
           end
       push!(getNet($(esc(m))).children, $(esc(node)))
           registermodel($(esc(m)), string($(quot(node))), $(esc(node)))
       end
    else
    error("not supported")
    end
end

macro addNode(m, node, nodename)
       return quote
           if haskey($(esc(node)).ext, :Net)
                getNet($(esc(node))).parent = $(esc(m))
           else
                $(esc(node)).ext[:Net] = NetData(JuMP.Model[], $(esc(m)),Dict{Symbol, JuMP.Model}())
           end
           push!(getNet($(esc(m))).children, $(esc(node)))
           registermodel($(esc(m)), $(esc(nodename)), $(esc(node)))
       end
end


# functions from this line are all only for stochastic problems
function getsumobjectivevalue(m::JuMP.Model)
     children = getchildren(m)
         leafModelList = children
         modelList = [m; leafModelList]
     objVal = 0
         for (idx,node) in enumerate(modelList)
        objVal += node.objVal
         end
     return objVal
end


function StochasticModel(createModel, data)
    nscen = length(data)
    master=NetModel()
    node1 = createModel(data[1])
    nfirst = length(node1.ext[:firstVarsId])
    @variable(master, first[1:nfirst])
    for i in 1:nscen
        node = createModel(data[i])
        @addNode(master, node, "s$i")
    firstVarsId = node.ext[:firstVarsId]
    for j = 1:nfirst
        if firstVarsId[j] > 0
            @constraint(master, first[j] == VariableRef(node, firstVarsId[j]))
        end
    end
    end
end

#0.5  2.0
function RandomStochasticModel(createModel, nscen=100, nfirst=5, nparam=5, rdl=0.0, rdu=2.0, adl=-10, adu=10 )
    srand(1234)
    master=NetModel()
    @variable(master, first[1:nfirst])
    firstVarsId = collect(1:nfirst)

    for i in 1:nscen
        node = createModel()
        @addNode(master, node, "s$i")
    firstVarsId = 1:nfirst
    node.ext[:firstVarsId] = firstVarsId
        for j = 1:nfirst
            if firstVarsId[j] > 0
                @constraint(master, first[j] == VariableRef(node, firstVarsId[j]))
            end
        end
    nmodified = 0
    if i == 1
        continue
    end
    
        if (typeof(nparam) == Int64) && nmodified < floor(nparam/2)
            for c = 1:length(node.quadconstr)
                con = node.quadconstr[c]
                terms = con.terms
                aff = terms.aff
                vars = [terms.qvars1;terms.qvars2; aff.vars]
                varsId = zeros(Int, length(vars))
                for k in 1:length(vars)
                    varsId[k] = vars[k].col
                end
                varsId = sort(union(varsId))
                if length(findin(varsId, firstVarsId))  == length(varsId)
                    continue
                end
                if typeof(nparam) == Int64
                    if nmodified >= nparam
                        break
                    end
                end

                if  con.sense == :(==)
                    connew = copy(con, node)
            connew.terms.aff.constant = addnoise(connew.terms.aff.constant, 0, adu, rdl, rdu)
            connew.sense = :(>=)
                    con.terms.aff.constant = addnoise(con.terms.aff.constant, adl, 0, -rdu, -rdl)
                    con.sense = :(<=)
                    push!(node.quadconstr, connew)
                elseif con.sense == :(>=)
                    aff.constant = addnoise(aff.constant, 0, adu, rdl, rdu)
                elseif con.sense == :(<=)
                    aff.constant = addnoise(aff.constant, adl, 0, -rdu, -rdl)
                end
        nmodified += 1
            end
        end
    

    for c in 1:length(node.linconstr)
        con = node.linconstr[c]
        terms = con.terms
        vars = terms.vars
        varsId = zeros(Int, length(vars))
        for k in 1:length(vars)
            varsId[k] = vars[k].col
        end
        varsId = sort(union(varsId))
        if length(findin(varsId, firstVarsId)) == length(varsId)
            continue
        end

        if typeof(nparam) == Int64
                if nmodified >= nparam
                    break
                end
        elseif typeof(nparam) == UnitRange{Int64}
            if !in(c, nparam)
                    continue
                end
        end

        if  con.lb == con.ub
            connew = copy(con, node)
            connew.lb = addnoise(connew.lb, adl, 0, -rdu, -rdl)
        connew.ub = Inf
                con.ub = addnoise(con.ub, 0, adu, rdl, rdu)
        con.lb = -Inf
        push!(node.linconstr, connew)
        elseif con.lb == -Inf
            con.ub = addnoise(con.ub, 0, adu, rdl, rdu)
        elseif con.ub  == -Inf
            con.lb = addnoise(con.lb, adl, 0, -rdu, -rdl)
            end
            nmodified += 1
    end

    #=
    if (typeof(nparam) == Int64) && nmodified < nparam
        for c = 1:length(node.quadconstr)
                con = node.quadconstr[c]
                terms = con.terms
        aff = terms.aff
                vars = [terms.qvars1;terms.qvars2; aff.vars]
                varsId = zeros(Int, length(vars))
                for k in 1:length(vars)
                    varsId[k] = vars[k].col
                end
        varsId = sort(union(varsId))
                if length(findin(varsId, firstVarsId))  == length(varsId)
                    continue
                end
        if typeof(nparam) == Int64
                    if nmodified >= nparam
                        break
                    end
        end
                if  con.sense == :(==)
                    connew = copy(con, node)
                    connew.terms.aff.constant = addnoise(connew.terms.aff.constant, 0, adu, rdl, rdu)
                    connew.sense = :(>=)
                    con.terms.aff.constant = addnoise(con.terms.aff.constant, adl, 0, -rdu, -rdl)
                    con.sense = :(<=)
                    push!(node.quadconstr, connew)
                elseif con.sense == :(>=)
                    aff.constant = addnoise(aff.constant, 0, adu, rdl, rdu)
                elseif con.sense == :(<=)
                    aff.constant = addnoise(aff.constant, adl, 0, -rdu, -rdl)
                end
        #aff.constant = addnoise(aff.constant, adl, adu, rdl, rdu)
        nmodified += 1
        end
    end
    =#

        for v in 1:node.numCols
            if typeof(nparam) == Int64
                if nmodified >= nparam
                    break
                end
            end
        if length(findin(v, firstVarsId)) != 0
            continue
        end
            node.colLower[v] = addnoise(node.colLower[v], adl, 0, -rdu, -rdl)
            node.colUpper[v] = addnoise(node.colUpper[v], 0, adu, rdl, rdu)
            nmodified += 1
        end

    if (i == 1) && (typeof(nparam) == Int64) && (nmodified < nparam)
        println("warning: the number of linear/quadratic second stage constraint  ", nmodified, " is less than nparam ",nparam)
    end
    end
    return master
end


function addnoise(a, adl, adu, rdl, rdu)
    if a == 0
        d = Uniform(adl, adu)
        a = a + rand(d)
    else
    d = Uniform(rdl, rdu)
    a = a + abs(a) * rand(d)
    end
    return a
end


function Base.copy(v::Array{VariableRef}, new_model::Model, indent::Int)
    ret = similar(v, VariableRef, size(v))
    for I in eachindex(v)
        ret[I] = VariableRef(new_model, v[I].col+indent)
    end
    ret
end

function Base.copy(v::Array{VariableRef}, new_model::Model, v_map::Array{Int, 1})
    ret = similar(v, VariableRef, size(v))
    for I in eachindex(v)
        ret[I] = VariableRef(new_model, v_map[v[I].col])
    end
    ret
end

#=
function extensiveModel(P::JuMP.Model)
        m = Model()
    m.ext[:v_map] = []
        children = getchildren(P)
        nscen = length(children)
        NLobj = false
        if P.nlpdata != nothing
            if P.nlpdata.nlobj != nothing
                NLobj = true
            end
        else
            for scen = 1:nscen
                node = children[scen]
                if node.nlpdata != nothing
                    if node.nlpdata.nlobj != nothing
                        NLobj = true
                        break
                    end
                end
            end
        end
        num_vars = P.numCols
        #add the node model variables to the new model
        for i = 1:num_vars
            x = JuMP.@variable(m)            #create an anonymous variable
            setlowerbound(x, P.colLower[i])
            setupperbound(x, P.colUpper[i])
            var_name = string(VariableRef(P,i))
            setname(x,var_name)                            #rename the variable to the node model variable name plus the node or edge name
            setcategory(x, P.colCat[i])                            #set the variable to the same category
            setvalue(x, P.colVal[i])                               #set the variable to the same value
        end
        m.obj = copy(P.obj, m)
        m.objSense = P.objSense
        for scen = 1:nscen
            modelname = "s$(scen)"
            node = children[scen]
            nodeobj = addnodemodel!(m, node, modelname)
            m.obj.qvars1 = [m.obj.qvars1; nodeobj.qvars1]
            m.obj.qvars2 = [m.obj.qvars2; nodeobj.qvars2]
            m.obj.qcoeffs = [m.obj.qcoeffs; nodeobj.qcoeffs]
            m.obj.aff.vars = [m.obj.aff.vars; nodeobj.aff.vars]
            m.obj.aff.coeffs = [m.obj.aff.coeffs; nodeobj.aff.coeffs]
            m.obj.aff.constant += nodeobj.aff.constant
        end
        m.linconstr  = [map(c->copy(c, m), P.linconstr); m.linconstr]
        children = getchildren(P)
        for i = 1:length(P.linconstr)
            Pcon = P.linconstr[i]
            Pvars = Pcon.terms.vars
            mcon = m.linconstr[i]
            mvars = mcon.terms.vars
            for (j, Pvar) in enumerate(Pvars)
                if (Pvar.m != P)
            scen = -1
                    for k in 1:length(children)
                        if Pvar.m == children[k]
                            scen = k
                        end
                    end
                    new_id =  m.ext[:v_map][scen][Pvar.col]
                    mvars[j] = VariableRef(m, new_id)
                end
            end
        end
        return m
end
=#

function extensiveSimplifiedModel(P::JuMP.Model)
    # Make a new model and add dictionary extensions
    m = Model()
    m.ext[:v_map] = Dict{VariableRef,VariableRef}()
    m.ext[:fstg_n] = Dict{String,VariableRef}()
    m.ext[:c_map] = Dict{ConstraintRef,ConstraintRef}()
    NLobj = false
    children = getchildren(P)
    nscen = length(children)
    
    # Add the node model variables to the new model. All first
    # stage variables are stored by name. If there is a second stage
    # variable which has the same name as a first stage variable, it
    # will be replaced by the single first stage copy.
    for var in JuMP.all_variables(P)
        new_x = JuMP.@variable(m)            #create an anonymous variable
        push!(m.ext[:v_map], var => new_x)
        var_name = JuMP.name(var)
        push!(m.ext[:fstg_n],var_name=>new_x)
        new_name = var_name
        JuMP.set_name(new_x,new_name)
        if JuMP.start_value(var) != nothing
            JuMP.set_start_value(new_x,JuMP.start_value(var))
        end
     end

    #OBJECTIVE FUNCTION (store expression on combinedd_nodes)
    if !(_has_nonlinear_obj(P))
        #AFFINE OR QUADTRATIC OBJECTIVE
        new_objective = _copy_objective(P,m.ext[:v_map])
        objSense = objective_sense(P)
        set_objective_function(m,new_objective)
        set_objective_sense(m,objSense)
    else
        #NONLINEAR OBJECTIVE
        print("sorry, currently not support nonlinear objective function, please formulate it as a constraint!")
    end
    
    # Get bunch of reference values to store in the extensive model
    children = getchildren(P)
    nscen = length(children)
    ncols = Array{Int}(undef,nscen+1)
    nlinconstrs = Array{Int}(undef,nscen+1)
    ncols[1] = num_variables(P)
    
    # Copy over the objective function from the master model.
    # Linearly combine objectives of master and node models
    masterObj = JuMP.objective_function(P)
    masterObjTerms = masterObj.terms
    objConst = masterObj.constant
    
    # Iterate over the scenario models and add to the extensive
    # form of the model.
    for scen in 1:nscen
        node = children[scen]
        modelname = "s$(scen)"
        nodeobj = addnodeSimplifiedmodel!(m, node, modelname, scen)
        
        # If the objective is a variable reference, create dictionary
        # with coefficient and variableref
        if nodeobj isa JuMP.VariableRef
            dict = OrderedDict{JuMP.VariableRef,Int64}()
            push!(dict,nodeobj => 1)
            merge!(masterObjTerms,dict)
        else
            objTerms = nodeobj.terms
            print(typeof(objTerms))
            objConst += nodeobj.constant
            merge!(masterObjTerms,nodeobj.terms)
        end
    end
    kv = masterObjTerms
    newObj = GenericAffExpr(objConst,kv)
    set_objective_function(m,newObj)
    set_objective_sense(m,MOI.MIN_SENSE)
    m.ext[:ncols] = ncols
    m.ext[:nlinconstrs] = nlinconstrs
    return m
end


function addnodeSimplifiedmodel!(m::JuMP.Model,node::JuMP.Model, nodename, scen)
        old_numCols = num_variables(m)
        num_vars = num_variables(node)
        constraint_types = JuMP.list_of_constraint_types(node)
        
        v_map = Array{Int}(undef,num_vars)        #this dict will map linear index of the node model variables to the new model JuMP variables {index => JuMP.VariableRef}
        #add the node model variables to the new model
        for (idx,var) in enumerate(JuMP.all_variables(node))
            fVarr = findall(x->x==name(var),collect(keys(m.ext[:fstg_n])))
            if length(fVarr) == 0
                new_x = JuMP.@variable(m)            #create an anonymous variable
                push!(m.ext[:v_map], var => new_x)
                var_name = JuMP.name(var)
                new_name = var_name*nodename
                JuMP.set_name(new_x,new_name)
                if JuMP.start_value(var) != nothing
                    JuMP.set_start_value(new_x,JuMP.start_value(var))
                end
            else
                fStgVar = m.ext[:fstg_n][name(var)]
                push!(m.ext[:v_map], var => fStgVar)
            end
        end
           
        for (func,set) in constraint_types
            constraint_refs = JuMP.all_constraints(node, func, set)
            for constraint_ref in constraint_refs
                constraint = JuMP.constraint_object(constraint_ref)
                new_constraint = _copy_constraint(constraint,m)
                if typeof(constraint.func) == VariableRef
                    continue
                elseif typeof(constraint.func) == GenericAffExpr{Float64,VariableRef}
                    vars = [name(var) for var in collect(keys(constraint.func.terms))]
                elseif typeof(constraint.func) == GenericQuadExpr{Float64,VariableRef}
                    varsa = [name(var.a) for var in collect(keys(constraint.func.terms))]
                    varsb = [name(var.b) for var in collect(keys(constraint.func.terms))]
                    vars=[varsa;varsb]
                else
                    print(typeof(constraint.func))
                end
                # if it is a first stage constraints (a constraints has only only first stage variables)
                #print(length(findall(x->x in vars,collect(keys(m.ext[:fstg_n])))))
                if length(findall(x->x in vars,collect(keys(m.ext[:fstg_n])))) == length(vars) && scen != 1
                    print("First stg constraint\n")
                    continue
                end
                new_ref= JuMP.add_constraint(m,new_constraint)
                #push!(m.ext[:c_map], [constraint_ref] => new_ref[1])
            end
        end

        #Copy the non-linear constraints to the new model
        nlp_initialized = false
        if node.nlp_data !== nothing
            d = JuMP.NLPEvaluator(node)         #Get the NLP evaluator object.  Initialize the expression graph
            MOI.initialize(d,[:ExprGraph])
            nlp_initialized = true
            num_cons = MathProgBase.numconstr(node)
            
            for i = 1:length(node_model.nlp_data.nlconstr)
                expr = MOI.constraint_expr(d,i)                         #this returns a julia expression
                if !(MathProgBase.isconstrlinear(d,i))    #if it's not a linear constraint
                    expr = MOI.constr_expr(d,i)  #this returns a julia expression
                    _splice_nonlinear_variables!(expr,node_model,reference_map)        #splice the variables from var_map into the expression
                    varsId = extractVarsId(expr)
                    if length(findall(x->x==varsId, firstVarsId))  == length(varsId) && scen != 1
                        continue
                    end
                    constraint_ref = JuMP.ConstraintRef(node,JuMP.NonlinearConstraintIndex(i),new_nl_constraint.shape)
                    reference_map[constraint_ref] = new_nl_constraint
                    _splice_nonlinear_variables!(expr,node,reference_map)        #splice the variables from var_map into the expression
                    new_nl_constraint = JuMP.add_NL_constraint(m,expr)      #raw expression input for non-linear constraint
                end
            end
        end
        
        nodeobj = _copy_objective(node,m.ext[:v_map])
        return nodeobj
end



#=
function addnodemodel!(m::JuMP.Model,node::JuMP.Model, nodename)
        old_numCols = m.numCols
        num_vars = node.numCols
        v_map = Array{Int}(num_vars)               #this dict will map linear index of the node model variables to the new model JuMP variables {index => JuMP.VariableRef}
        #add the node model variables to the new model
        for i = 1:num_vars
            x = JuMP.@variable(m)            #create an anonymous variable
            setlowerbound(x, node.colLower[i])
            setupperbound(x, node.colUpper[i])
            var_name = string(VariableRef(node,i))
            setname(x,nodename*var_name)                             #rename the variable to the node model variable name plus the node or edge name
            setcategory(x, node.colCat[i])                            #set the variable to the same category
            setvalue(x,node.colVal[i])                               #set the variable to the same value
            v_map[i] = x.col                                         #map the linear index of the node model variable to the new variable
        end
        
        push!(m.ext[:v_map], v_map)
        greaterThan = all_constraints(node,GenericAffExpr{Float64,VariableRef},MOI.GreaterThan{Float64})
        equalTo = all_constraints(node,GenericAffExpr{Float64,VariableRef},MOI.EqualTo{Float64})
        lessThan = all_constraints(node,GenericAffExpr{Float64,VariableRef},MOI.LessThan{Float64})
        constraintTypes = [greaterThan,equalTo,lessThan]
        
        for type in constraintTypes
        if len(type)>=1
        for i = 1:length(type)
            con = copy(type, node)
            con.terms.vars = copy(con.terms.vars, m, v_map)
            m.linconstr = [m.linconstr; con]
        end
        end
        end
        
    for i = 1:length(node.quadconstr)
            con = copy(node.quadconstr[i], node)
        terms = con.terms
            terms.qvars1 = copy(terms.qvars1, m, old_numCols)
            terms.qvars2 = copy(terms.qvars2, m, old_numCols)
            terms.aff.vars = copy(terms.aff.vars, m, old_numCols)
        m.quadconstr = [m.quadconstr; con]
    end
        #Copy the non-linear constraints to the new model
        d = JuMP.NLPEvaluator(node)         #Get the NLP evaluator object.  Initialize the expression graph
        MathProgBase.initialize(d,[:ExprGraph])
        num_cons = MathProgBase.numconstr(node)
        for i = (1+length(node.linconstr)+length(node.quadconstr)):num_cons
            if !(MathProgBase.isconstrlinear(d,i))    #if it's not a linear constraint
                expr = MathProgBase.constr_expr(d,i)  #this returns a julia expression
                _splicevars!(expr,m, v_map)              #splice the variables from v_map into the expression
                addNLconstraint2(m,expr)          #raw expression input for non-linear constraint
            end
        end
        nodeobj = copy(node.obj)
        nodeobj.qvars1 = copy(nodeobj.qvars1, m, old_numCols)
        nodeobj.qvars2 = copy(nodeobj.qvars2, m, old_numCols)
        nodeobj.aff.vars = copy(nodeobj.aff.vars, m, old_numCols)
        return nodeobj
end
=#

function copyNLModel(P::JuMP.Model)  # copy model and convert quadratic consriants and obj to nonlinear
    m = Model()
    # Variables
    m.numCols = P.numCols
    m.colNames = P.colNames[:]
    m.colNamesIJulia = P.colNamesIJulia[:]
    m.colLower = P.colLower[:]
    m.colUpper = P.colUpper[:]
    m.colCat = P.colCat[:]
    m.colVal = P.colVal[:]
    m.linconstrDuals = P.linconstrDuals[:]
    m.redCosts = P.redCosts[:]
    # Constraints
    m.linconstr  = map(c->copy(c, m), P.linconstr)
    #m.quadconstr = map(c->copy(c, m), P.quadconstr)

    m.objDict = Dict{Symbol,Any}()
    m.varData = ObjectIdDict()
    for (symb,o) in P.objDict
        newo = copy(o, m)
        m.objDict[symb] = newo
        if haskey(P.varData, o)
            m.varData[newo] = P.varData[o]
        end
    end

    # Objective
    m.obj = copy(P.obj, m)
    m.objSense = P.objSense
    #Copy the non-linear constraints to the new model
    d = JuMP.NLPEvaluator(P)     #Get the NLP evaluator object.  Initialize the expression graph
    MathProgBase.initialize(d,[:ExprGraph])
    num_cons = MathProgBase.numconstr(P)
    for i = (1+length(P.linconstr)):num_cons
            expr = MathProgBase.constr_expr(d,i)  #this returns a julia expression
        _splicevars!(expr, m)
            addNLconstraint2(m,expr)          #raw expression input for non-linear constraint
    end
    if !MathProgBase.isobjlinear(d)  #isa(P.nlpdata.nlobj, NonlinearExprData) || MathProgBase.isobjquadratic(d)  #P.nlpdata.nlobj != nothing
        expr = MathProgBase.obj_expr(d)
    _splicevars!(expr, m)
        JuMP.setNLobjective(m, P.objSense, expr)
    end
    if !isempty(P.ext)
        m.ext = similar(P.ext)
        for (key, val) in P.ext
            m.ext[key] = try
                copy(P.ext[key])
            catch
                continue;  #error("Error copying extension dictionary. Is `copy` defined for all your user types?")
            end
        end
    end
    return m
end


function copyStoModel(P::JuMP.Model)
    if haskey(P.ext, :Net)
        m = NetModel()
    children = P.ext[:Net].children[:]
    nscen = length(children)
        for scen = 1:nscen
        modelname = "s$(scen)"
            nodecopy,_= copyModel(children[scen])
        @addNode(m, nodecopy, modelname)
        end
    m,r = copyModel(P)
    end
    return m
end


#splice variables into a constraint expression
function _splicevars!(expr::Expr, m::JuMP.Model, v_map=nothing)
if v_map !=nothing
    for i = 1:length(expr.args)
        if typeof(expr.args[i]) == Expr
            if expr.args[i].head != :ref   #keep calling _splicevars! on the expression until it's a :ref. i.e. :(x[index])
                _splicevars!(expr.args[i], m, v_map)
            else  #it's a variable
                var_index = expr.args[i].args[2]   #this is the actual index in x[1], x[2], etc...
                new_var = :($(VariableRef(m, v_map[var_index])))   #get the JuMP variable from v_map using the index
                expr.args[i] = new_var             #replace :(x[index]) with a :(JuMP.VariableRef)
            end
        end
    end
else
    for i = 1:length(expr.args)
        if typeof(expr.args[i]) == Expr
            if expr.args[i].head != :ref   #keep calling _splicevars! on the expression until it's a :ref. i.e. :(x[index])
                _splicevars!(expr.args[i], m, v_map)
            else  #it's a variable
                var_index = expr.args[i].args[2]   #this is the actual index in x[1], x[2], etc...
                new_var = :($(VariableRef(m, var_index)))   #get the JuMP variable from v_map using the index
                expr.args[i] = new_var             #replace :(x[index]) with a :(JuMP.VariableRef)
            end
        end
    end
end
end


function extractVarsId(expr::Expr)
    varsId = Int[]
    if expr.head == :ref
       var_index = expr.args[2]   #this is the actual index in x[1], x[2], etc...
       return [var_index]
    end
    for i = 1:length(expr.args)
        if typeof(expr.args[i]) == Expr
            varsId = [varsId; extractVarsId(expr.args[i])]
        end
    end
    return union(varsId)
end

function extractVarsId(vars::Array{VariableRef,1})
    varsId = Int[]
    for i = 1:length(vars)
        push!(varsId, vars[i].index.value)
    end
    return union(varsId)
end

function extractVarsId(vars::Array{UnorderedPair{VariableRef},1})
    varsId = Int[]
    for i = 1:length(vars)
        push!(varsId, vars[i].a.index.value)
        push!(varsId, vars[i].b.index.value)
    end
    return union(varsId)
end


function removeConnection(master::JuMP.Model)
    deleterow = []
    for row in 1:length(master.linconstr)
        coeffs = master.linconstr[row].terms.coeffs
        vars   = master.linconstr[row].terms.vars
        for (it,ind) in enumerate(coeffs)
            if (vars[it].m) != master
                push!(deleterow, row)
                break
            end
        end
    end
    deleted = master.linconstr[deleterow]
    deleteat!(master.linconstr,deleterow)
    return deleted
end


# args[3] is sure 0 #a number (prabably 0.0)
# Ex: addNLconstraint(m, :($x + $y^2 <= 1))
function addNLconstraint2(m::Model, ex::Expr)
     @assert ex.args[3] == 0.0
     @assert JuMP.isexpr(ex, :call)
     #=
     if (ex.args[3] != 0.0) || !JuMP.isexpr(ex, :call)
          return addNLconstraint(m, ex)
     end
     =#
    JuMP.initNLP(m)
    m.internalModelLoaded = false
        # Simple comparison - move everything to the LHS
        op = ex.args[1]
        if op == :(==)
            lb = 0.0
            ub = 0.0
        elseif op == :(<=) || op == :(≤)
            lb = -Inf
            ub = 0.0
        elseif op == :(>=) || op == :(≥)
            lb = 0.0
            ub = Inf
        else
            error("in addNLconstraint ($ex): expected comparison operator (<=, >=, or ==).")
        end
        lhs = :($(ex.args[2]))
        c = JuMP.NonlinearConstraint(JuMP.NonlinearExprData(m, lhs), lb, ub)
        push!(m.nlpdata.nlconstr, c)
        return ConstraintRef{Model,NonlinearConstraint}(m, length(m.nlpdata.nlconstr))
end

end
#include("NetParPipsNlp.jl")
#include("NetIpopt.jl")
#end
